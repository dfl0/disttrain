"""Synchronous parameter servers: edges sum contributions, only the root updates."""
import time
import numpy as np
import zmq

from protocol import (Msg, MAX_VERSION, decode_packet, encode_packet, encode_assignment,
                      decode_assignment, encode_version, decode_version, encode_evaluation)
from transport import send_message, receive_message, close_sockets, drain_socket
from timetree import TimeTree


class ParameterServer:
    synchronous = True
    gradient_message = Msg.PUSH_GRADS
    snapshot_message = Msg.PUSH_PARAMS
    done_message = Msg.DONE
    def __init__(self, model, children_ids, sync, router_sock, serializer, init_params,
                 *, child_ranks, parent=None, dealer_sock=None, stop_sock=None,
                 eval_push_sock=None, lr=0.01, eval_interval=200, global_state=None,
                 log=print, node_id=None, lifecycle=None, max_rounds=None, log_interval=10, recorder=None, worker_count=None):
        if sync is not self.synchronous:
            raise ValueError('server implementation does not match configured mode')
        if not children_ids or len(set(children_ids)) != len(children_ids):
            raise ValueError('server requires nonempty unique children')
        if set(child_ranks) != set(children_ids):
            raise ValueError('assignments must cover exactly the configured children')
        if any(type(rank) is not int or rank < -1 for rank in child_ranks.values()):
            raise ValueError('child assignments must be global ranks or -1 for edges')
        if parent is not None and (node_id is None or dealer_sock is None):
            raise ValueError('edge requires a node ID and parent socket')
        if not np.isfinite(lr) or lr <= 0 or type(eval_interval) is not int or eval_interval <= 0:
            raise ValueError('learning rate and evaluation interval must be positive')
        self.model, self._serializer, self._init_params_fn = model, serializer, init_params
        self.children_ids, self.child_ranks = tuple(children_ids), dict(child_ranks)
        self.parent, self.node_id = parent, node_id
        self.is_root = parent is None
        if max_rounds is not None and (type(max_rounds) is not int or max_rounds <= 0):
            raise ValueError("max_rounds must be a positive integer or None")
        if type(log_interval) is not int or log_interval <= 0:
            raise ValueError("log_interval must be a positive integer")
        self.log_interval = log_interval
        self.recorder = recorder
        self.worker_count = worker_count or len(children_ids)
        self.applied_contributions = 0
        self.update_elapsed = 0.0
        self.total_samples = 0
        self.training_started = time.perf_counter()
        self.last_logged_version = -1
        self.max_rounds = max_rounds
        self.lr, self.eval_interval = lr, eval_interval
        self.lifecycle = lifecycle
        self.send = send_message if lifecycle is None else lifecycle.send
        self.receive = receive_message if lifecycle is None else lifecycle.receive
        self.log = log
        self.global_state = {} if global_state is None else global_state
        self._router, self._dealer = router_sock, dealer_sock
        self._stop_sock, self._eval_push = stop_sock, eval_push_sock
        self.flat_params = np.empty(serializer.n_params, dtype=np.float32)
        serializer.flatten_params_into(self.flat_params)
        self.gradient_sum = np.zeros_like(self.flat_params)
        self.sample_count = 0
        self.version = 0
        self.received = set()
        self.finished = set()
        self.registered = set()
        self.waiting_for_parent = False
        self.stopped = False
        self.timer = TimeTree(print_func=log)

    def run(self):
        try:
            if self.lifecycle:
                self.lifecycle.ready()
            self._init_params()
            if self.lifecycle:
                self.lifecycle.begin_round()
            self.training_started = time.perf_counter()
            if self.is_root:
                self.log(f'[TRAIN] ready; waiting for first {"round" if self.synchronous else "update"}')
            self._event_loop()
        finally:
            self._shutdown()

    def _init_params(self):
        if self.is_root:
            self.model.apply(self._init_params_fn)
            self._serializer.flatten_params_into(self.flat_params)
        else:
            self.send(self._dealer, Msg.REGISTER, self.node_id.encode())
            _, kind, payload = self.receive(self._dealer)
            if kind != bytes([Msg.ASSIGN]) or decode_assignment(payload) != -1:
                raise ValueError('edge expected its registration acknowledgement')
            _, kind, payload = self.receive(self._dealer)
            if kind != bytes([self.snapshot_message]):
                raise ValueError('edge expected initial root snapshot')
            version, params = self._decode_initial_snapshot(payload)
            if version != 0:
                raise ValueError('initial snapshot must be version zero')
            self.flat_params[:] = params
        while len(self.registered) < len(self.children_ids):
            identity, kind, payload = self.receive(self._router, routed=True)
            if kind != bytes([Msg.REGISTER]) or identity != payload:
                raise ValueError('registration type or identity/payload mismatch')
            child_id = identity.decode('utf-8')
            if child_id not in self.child_ranks or child_id in self.registered:
                raise ValueError(f'unknown or duplicate registration: {child_id}')
            self.registered.add(child_id)
            self.send(self._router, Msg.ASSIGN, encode_assignment(self.child_ranks[child_id]), identity=identity)
        self._broadcast_snapshot()

    def _decode_initial_snapshot(self, payload):
        version, _, params = decode_packet(payload, self.flat_params.size)
        return version, params

    def _event_loop(self):
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        control = self._stop_sock if self.is_root else self._dealer
        if control is not None:
            poller.register(control, zmq.POLLIN)
        with self.timer.measure('train'):
            while not self.stopped:
                sockets = (self.lifecycle.wait([s for s in (self._router, control) if s is not None])
                           if self.lifecycle else dict(poller.poll()))
                # Observe termination/parent response before more child traffic.
                if control in sockets:
                    _, kind, payload = self.receive(control)
                    if kind == bytes([Msg.STOP]):
                        self._dispatch_stop()
                        break
                    if self.is_root or kind != bytes([self.snapshot_message]):
                        raise ValueError('unexpected control message')
                    self._dispatch_parent_push(payload)
                if not self.stopped and self._router in sockets:
                    identity, kind, payload = self.receive(self._router, routed=True)
                    child_id = identity.decode('utf-8')
                    if child_id not in self.registered:
                        raise ValueError('message from an unregistered child')
                    if kind == bytes([self.gradient_message]):
                        self._accept_contribution(child_id, payload)
                    elif kind == bytes([self.done_message]):
                        self._accept_done(child_id, payload)
                    else:
                        raise ValueError('unexpected child message')

    def _accept_contribution(self, child_id, payload):
        if (self.stopped or child_id not in self.registered or child_id in self.received
                or self.finished or self.waiting_for_parent):
            raise ValueError('unknown, duplicate or out-of-phase contribution')
        version, count, gradients = decode_packet(payload, self.flat_params.size, gradients=True)
        if version != self.version or self.sample_count + count > MAX_VERSION:
            raise ValueError('wrong round/version or sample-count overflow')
        # Validate the prospective sum before committing any round state.
        with np.errstate(over='ignore', invalid='ignore'):
            combined = self.gradient_sum + gradients
        if not np.isfinite(combined).all():
            raise ValueError('gradient sum overflow')
        self.gradient_sum[:] = combined
        self.sample_count += count
        self.received.add(child_id)
        if len(self.received) == len(self.children_ids):
            if self.is_root:
                self._commit_root_round()
            else:
                self._forward_edge_round()

    def _commit_root_round(self):
        if self.version == MAX_VERSION:
            raise ValueError('root version overflow')
        with np.errstate(over='ignore', invalid='ignore'):
            updated = self.flat_params - self.lr * (self.gradient_sum / self.sample_count)
        if not np.isfinite(updated).all():
            raise ValueError('non-finite root update')
        self.flat_params[:] = updated
        self.version += 1
        self.total_samples += self.sample_count
        self.applied_contributions += self.worker_count
        self.update_elapsed = time.perf_counter() - self.training_started
        if self.recorder:
            self.recorder.update(self.update_elapsed, self.version, self.total_samples,
                                 self.applied_contributions, self.sample_count, self.worker_count, 0, 0)
        if self.version == 1 or self.version % self.log_interval == 0:
            self._log_progress()
        self._reset_round()
        if self.max_rounds is not None and self.version >= self.max_rounds:
            self.log(f'max_rounds reached at version {self.version}; evaluating final model')
            self._finish_training()
            return  # Do not release another round while final evaluation runs.
        self._broadcast_snapshot()
        if self.version % self.eval_interval == 0:
            self._push_eval()

    def _forward_edge_round(self):
        self.send(self._dealer, Msg.PUSH_GRADS,
                     encode_packet(self.version, self.gradient_sum, self.sample_count))
        self.waiting_for_parent = True
        # Retain the completed round until the matching parent response arrives.

    def _dispatch_parent_push(self, payload):
        version, _, params = decode_packet(payload, self.flat_params.size)
        if self.is_root or not self.waiting_for_parent or self.finished or version != self.version + 1:
            raise ValueError('unexpected parent snapshot or wrong version')
        self.flat_params[:] = params
        self.version = version
        self.waiting_for_parent = False
        self._reset_round()
        self._broadcast_snapshot()

    def _reset_round(self):
        self.gradient_sum.fill(0)
        self.sample_count = 0
        self.received.clear()

    def _broadcast_snapshot(self):
        if self.lifecycle and self.version > 0:
            self.lifecycle.begin_round()
        payload = encode_packet(self.version, self.flat_params)
        for child_id in self.children_ids:
            self.send(self._router, Msg.PUSH_PARAMS, payload, identity=child_id.encode())

    def _accept_done(self, child_id, payload):
        if (self.stopped or child_id not in self.registered or child_id in self.finished
                or self.received or self.waiting_for_parent or decode_version(payload) != self.version):
            raise ValueError('invalid completion or unequal worker round counts')
        self.finished.add(child_id)
        if len(self.finished) == len(self.children_ids):
            if self.is_root:
                self._finish_training()
            else:
                self.send(self._dealer, Msg.DONE, encode_version(self.version))
                self.waiting_for_parent = True  # only STOP is valid now

    def _log_progress(self):
        elapsed = time.perf_counter() - self.training_started
        self.log(f'[TRAIN] round={self.version} samples={self.total_samples} elapsed={elapsed:.2f}s')
        self.last_logged_version = self.version

    def _finish_training(self):
        if self.last_logged_version != self.version:
            self._log_progress()
        self._push_eval()
        if self.lifecycle:
            self.lifecycle.wait_evaluation(self.version)
        self._dispatch_stop()

    def _push_eval(self):
        if self._eval_push is not None:
            self.send(self._eval_push, Msg.EVAL, encode_evaluation(self.version, self.total_samples, self.update_elapsed, self.flat_params))

    def _dispatch_stop(self):
        self.stopped = True
        if self.lifecycle:
            return  # Runtime coordinates STOP/ACK over the independent control channel.
        for child_id in self.children_ids:
            self.send(self._router, Msg.STOP, identity=child_id.encode())
        if self.is_root and self._eval_push is not None:
            self.send(self._eval_push, Msg.STOP)

    def _shutdown(self):
        # Existing best-effort transport cleanup; acknowledged shutdown is separate.
        if self.lifecycle is None:
            drain_socket(self._router, seconds=2.0)
        close_sockets(self._router, self._dealer, self._stop_sock, self._eval_push)
