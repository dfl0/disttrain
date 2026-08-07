import time
import numpy as np
import zmq

from comm import Msg
from timetree import TimeTree


class ParameterServer:
    """
    Parameter server for both root and edge (intermediate) roles.

    Pass parent=None to create a root PS. Pass a parent address string to create
    an edge PS. Uses ZMQ ROUTER (facing children) and optionally DEALER (facing
    parent PS) sockets.

    Call run() to enter the training loop. It blocks until a STOP signal
    propagates through the tree.
    """

    def __init__(
        self,
        model,
        children_hostnames,
        sync,
        router_sock,
        serializer,
        init_params,
        parent=None,
        dealer_sock=None,
        stop_sock=None,
        eval_push_sock=None,
        lr=0.01,
        eval_interval=200,
        global_state=None,
        log=None,
    ):
        self._serializer     = serializer
        self._init_params_fn = init_params

        self.model              = model
        self.children_hostnames = children_hostnames  # ordered list of child hostnames
        self.sync               = sync
        self.parent             = parent
        self.lr                 = lr
        self.eval_interval      = eval_interval
        self.global_state       = global_state or {}
        self.log                = log or print

        self.is_root = (parent is None)

        # ZMQ sockets
        self._router    = router_sock
        self._dealer    = dealer_sock       # edge only
        self._stop_sock = stop_sock         # root only (PULL from evaluator)
        self._eval_push = eval_push_sock    # root only (PUSH to evaluator)

        # parameter buffers
        self.flat_params  = np.empty(serializer.n_params, dtype=np.float32)
        serializer.flatten_params_into(self.flat_params)
        self.packet_size  = serializer.n_params + 1
        self._send_buf    = np.empty(self.packet_size, dtype=np.float32)
        # anchor_params: edge PS tracks param state at start of each aggregation window
        self.anchor_params = np.empty_like(self.flat_params)

        # identity maps — populated during registration in _init_params
        self._identity_to_idx = {}   # zmq identity bytes → child_idx (int)
        self._idx_to_identity = {}   # child_idx (int) → zmq identity bytes

        # counters and state
        self.model_version   = 0
        self.n_updates       = 0
        self.active_children = len(children_hostnames)
        self.children_done   = set()
        self.eval_stopped    = False

        # precomputed update scaling and scratch buffer
        self._lr_scale   = lr / max(1, len(children_hostnames))
        self._update_buf = np.empty_like(self.flat_params)

        self.timer = TimeTree(print_func=self.log)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        self.log(f"[{'SYNC' if self.sync else 'ASYNC'} {'PS' if self.is_root else f'EDGE PS, parent: {self.parent}'}]")
        self._init_params()
        self.log("Starting training...")
        self._event_loop()
        self._shutdown()
        self.timer.print_tree()

    # ------------------------------------------------------------------
    # Setup: registration and initial parameter broadcast
    # ------------------------------------------------------------------

    def _init_params(self):
        if self.is_root:
            self.model.apply(self._init_params_fn)
            self._serializer.flatten_params_into(self.flat_params)
            self.model_version = 0
        else:
            # register with parent and receive initial params
            self._dealer.send_multipart([
                bytes([Msg.REGISTER]),
                _hostname().encode(),
            ])
            # wait for ASSIGN (worker_idx, not used by PS but kept for protocol symmetry)
            frames = self._dealer.recv_multipart()
            # frames: [msg_type, worker_idx_bytes]
            # then wait for initial PUSH_PARAMS
            frames = self._dealer.recv_multipart()
            assert frames[0] == bytes([Msg.PUSH_PARAMS])
            packet = np.frombuffer(frames[1], dtype=np.float32)
            self.model_version = int(packet[0])
            self.flat_params[:] = packet[1:]
            self.anchor_params[:] = self.flat_params

        # wait for all children to register
        registered = {}
        while len(registered) < len(self.children_hostnames):
            frames = self._router.recv_multipart()
            identity, msg_type = frames[0], frames[1]
            hostname = frames[2].decode() if len(frames) > 2 else ""
            assert msg_type == bytes([Msg.REGISTER]), f"expected REGISTER, got {msg_type}"
            idx = self.children_hostnames.index(hostname)
            registered[identity] = idx
            self._router.send_multipart([
                identity,
                bytes([Msg.ASSIGN]),
                np.int32(idx).tobytes(),
            ])

        self._identity_to_idx = registered
        self._idx_to_identity = {v: k for k, v in registered.items()}

        # send initial params to all registered children
        self._pack_send_buf()
        payload = self._send_buf.tobytes()
        for identity in self._identity_to_idx:
            self._router.send_multipart([identity, bytes([Msg.PUSH_PARAMS]), payload])

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _event_loop(self):
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        if not self.is_root:
            poller.register(self._dealer, zmq.POLLIN)
        if self.is_root:
            poller.register(self._stop_sock, zmq.POLLIN)

        self.timer.reset()
        with self.timer.measure("train"):
            while True:
                with self.timer.measure("wait") as t:
                    socks = dict(poller.poll())

                if self._router in socks:
                    frames = self._router.recv_multipart()
                    identity  = frames[0]
                    msg_type  = frames[1]
                    payload   = frames[2] if len(frames) > 2 else b''

                    if msg_type == bytes([Msg.PUSH_GRADS]):
                        child_idx = self._identity_to_idx[identity]
                        t["tag"] = self.children_hostnames[child_idx]
                        self._dispatch_child_grad(child_idx, identity, payload)
                        self._update_global_state()

                if not self.is_root and self._dealer in socks:
                    frames    = self._dealer.recv_multipart()
                    msg_type  = frames[0]
                    payload   = frames[1] if len(frames) > 1 else b''

                    if msg_type == bytes([Msg.PUSH_PARAMS]):
                        t["tag"] = self.parent
                        self._dispatch_parent_push(payload)
                        self._update_global_state()
                    elif msg_type == bytes([Msg.STOP]):
                        t["tag"] = "stop"
                        self._dispatch_stop()
                        break

                if self.is_root and self._stop_sock in socks:
                    self._stop_sock.recv()   # consume; payload unused
                    t["tag"] = "stop"
                    self._dispatch_stop()
                    break

    # ------------------------------------------------------------------
    # Dispatch handlers
    # ------------------------------------------------------------------

    def _dispatch_stop(self):
        self.eval_stopped = True
        self.log("Stop signal received, propagating to children")
        for identity in self._identity_to_idx:
            self._router.send_multipart([identity, bytes([Msg.STOP]), b''])

    def _dispatch_parent_push(self, payload):
        packet = np.frombuffer(payload, dtype=np.float32)
        self.model_version = int(packet[0])
        self.flat_params[:] = packet[1:]
        self.anchor_params[:] = self.flat_params

        if self.sync:
            self._pack_send_buf()
            wire_payload = self._send_buf.tobytes()
            with self.timer.measure("send_params") as ts:
                for identity, child_idx in self._identity_to_idx.items():
                    self._router.send_multipart([identity, bytes([Msg.PUSH_PARAMS]), wire_payload])
                    ts["tag"] = self.children_hostnames[child_idx]

    def _dispatch_child_grad(self, child_idx, identity, payload):
        packet = np.frombuffer(payload, dtype=np.float32)
        grad_version = int(packet[0])
        grads        = packet[1:]

        staleness = self.model_version - grad_version
        if staleness > 0:
            child_host = self.children_hostnames[child_idx]
            self.log(f"stale grad from {child_host}: version {grad_version}, current {self.model_version}, staleness {staleness}")

        np.multiply(grads, self._lr_scale, out=self._update_buf)
        self.flat_params -= self._update_buf
        self.n_updates   += 1

        self._apply_sync_or_async(child_idx, identity, grad_version)

    # ------------------------------------------------------------------
    # Update policy
    # ------------------------------------------------------------------

    def _apply_sync_or_async(self, child_idx, identity, grad_version):
        if self.sync:
            self.children_done.add(child_idx)
            if len(self.children_done) < self.active_children:
                return

            # all children have reported — commit the round
            self._increment_version()
            self._pack_send_buf()
            wire_payload = self._send_buf.tobytes()

            with self.timer.measure("send_params") as ts:
                for done_idx in self.children_done:
                    ident = self._idx_to_identity[done_idx]
                    self._router.send_multipart([ident, bytes([Msg.PUSH_PARAMS]), wire_payload])
                    ts["tag"] = self.children_hostnames[done_idx]

            if not self.is_root:
                self._send_delta_to_parent()

            self.children_done.clear()

            if self.is_root:
                self._maybe_push_eval()

        else:  # async
            self._increment_version()
            self._pack_send_buf()

            with self.timer.measure("send_params") as ts:
                self._router.send_multipart([identity, bytes([Msg.PUSH_PARAMS]), self._send_buf.tobytes()])
                ts["tag"] = self.children_hostnames[child_idx]

            if self.is_root:
                self._maybe_push_eval()
            else:
                if self.n_updates >= max(1, self.active_children // 2):
                    self._send_delta_to_parent()
                    self.anchor_params[:] = self.flat_params
                    self.n_updates = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _increment_version(self):
        self.model_version += 1
        assert self.model_version < 2**24, \
            f"model_version {self.model_version} exceeds float32 precision"

    def _pack_send_buf(self):
        self._send_buf[0] = self.model_version
        self._send_buf[1:] = self.flat_params

    def _send_delta_to_parent(self):
        self._send_buf[0] = self.model_version
        np.subtract(self.anchor_params, self.flat_params, out=self._send_buf[1:])
        with self.timer.measure("send_params") as ts:
            self._dealer.send_multipart([bytes([Msg.PUSH_GRADS]), self._send_buf.tobytes()])
            ts["tag"] = self.parent

    def _maybe_push_eval(self):
        if self.eval_stopped:
            return
        if self.model_version % self.eval_interval != 0:
            return
        self.log(f"sending params to evaluator, version: {self.model_version}")
        with self.timer.measure("send_params") as ts:
            self._eval_push.send_multipart([bytes([Msg.EVAL]), self._send_buf.tobytes()])
            ts["tag"] = "eval"

    def _update_global_state(self):
        # no send-queue depth to report with ZMQ; keep dict keys for metrics thread compatibility
        gs = self.global_state
        gs["n_child_send_reqs"]  = 0
        gs["n_parent_send_reqs"] = 0
        gs["send_reqs_children"] = []

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        role_name = "root ps" if self.is_root else "edge ps"
        self.log(f"{role_name} terminating...")

        # drain any residual PUSH_GRADS with a 2s timed window
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            remaining_ms = int((deadline - time.perf_counter()) * 1000)
            if not poller.poll(timeout=max(0, remaining_ms)):
                break
            self._router.recv_multipart()

        # set LINGER=0 on all sockets so context.term() doesn't hang
        for sock in filter(None, [self._router, self._dealer,
                                   self._stop_sock, self._eval_push]):
            sock.setsockopt(zmq.LINGER, 0)
            sock.close()

        self.log(f"{role_name} done cleaning up")


def _hostname():
    import socket
    return socket.gethostname()
