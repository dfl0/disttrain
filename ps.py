import numpy as np
from mpi4py import MPI

from comm import Msg, SendQueue, cancel_recv_req, safe_wait_send_req
from timetree import TimeTree


class ParameterServer:
    """
    Parameter server for both root (rank 0) and edge (intermediate) roles.

    Pass parent=None to create a root PS. Pass a parent rank to create an edge PS.
    The root watches workers/edges below it and an evaluator above it; the edge
    watches workers below it and its parent PS above it.

    Call run() to enter the training loop. It blocks until a STOP signal
    propagates through the tree.
    """

    def __init__(
        self,
        model,
        children,
        comm,
        sync,
        serializer,
        init_params,
        parent=None,
        evaluator=None,
        lr=0.01,
        eval_interval=200,
        global_state=None,
        log=None,
    ):
        self._serializer     = serializer
        self._init_params_fn = init_params

        self.model          = model
        self.children       = children
        self.comm           = comm
        self.sync           = sync
        self.parent         = parent
        self.evaluator      = evaluator
        self.lr             = lr
        self.eval_interval  = eval_interval
        self.global_state   = global_state or {}
        self.log            = log or print

        self.is_root = (parent is None)

        # parameter buffer
        self.flat_params   = np.empty(serializer.n_params, dtype=np.float32)
        serializer.flatten_params_into(self.flat_params)
        self.packet_size   = serializer.n_params + 1
        self.stop_buf      = np.empty(1, dtype=np.int32)
        self.child_packets = [np.empty(self.packet_size, dtype=np.float32) for _ in children]
        self.parent_packet = None if self.is_root else np.empty(self.packet_size, dtype=np.float32)
        # anchor_params tracks the parameter state at the start of each aggregation window (edge only)
        self.anchor_params = np.empty_like(self.flat_params)

        # send queues — pre-allocate buffer pools sized to packet_size
        buf_shape       = (self.packet_size,)
        child_inflight  = max(8, 4 * len(children))
        parent_inflight = max(4, 2 * len(children))
        self.child_sq  = SendQueue(comm, child_inflight,  buf_shape=buf_shape)
        self.parent_sq = None if self.is_root else SendQueue(comm, parent_inflight, buf_shape=buf_shape)

        # MPI request handles (populated in _init_params)
        self.child_recv_reqs = []
        self.stop_recv_req   = None
        self.parent_recv_req = None  # edge only

        # pre-allocated wait list — layout: [child_0..child_N, (parent_req,) stop_req]
        # indices computed once in _init_params after all requests are posted
        self._wait_list   = []
        self._stop_idx    = 0
        self._parent_idx  = 0  # edge only

        # counters and state
        self.model_version   = 0
        self.n_updates       = 0
        self.active_children = len(children)
        self.children_done   = set()
        self.eval_stopped    = False

        # pre-allocated stop signal buffers — one per child, all in-flight simultaneously at shutdown
        self._stop_bufs  = [np.zeros(1, dtype=np.int32) for _ in children]
        self.stop_reqs   = []  # (req, buf) pairs populated by _dispatch_stop

        # precomputed update scaling and scratch buffer to avoid per-update temp array
        self._lr_scale   = lr / max(1, len(children))
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
    # Setup
    # ------------------------------------------------------------------

    def _init_params(self):
        if self.is_root:
            self.model.apply(self._init_params_fn)
            self._serializer.flatten_params_into(self.flat_params)
            self.model_version = 0
            for child in self.children:
                buf = self._claim_param_buf(self.child_sq)
                self.child_sq.post(buf, dest=child, tag=Msg.PUSH_PARAMS)
            self.stop_recv_req = self.comm.Irecv(self.stop_buf, source=self.evaluator, tag=Msg.STOP)
        else:
            self.comm.Recv(self.parent_packet, source=self.parent, tag=Msg.PUSH_PARAMS)
            self.model_version = int(self.parent_packet[0])
            self.flat_params[:] = self.parent_packet[1:]
            self.anchor_params[:] = self.flat_params
            for child in self.children:
                self.child_sq.send(self.parent_packet, dest=child, tag=Msg.PUSH_PARAMS)
            self.parent_recv_req = self.comm.Irecv(self.parent_packet, source=self.parent, tag=Msg.PUSH_PARAMS)
            self.stop_recv_req   = self.comm.Irecv(self.stop_buf, source=self.parent, tag=Msg.STOP)

        self.child_recv_reqs = [
            self.comm.Irecv(self.child_packets[i], source=self.children[i], tag=Msg.PUSH_GRADS)
            for i in range(len(self.children))
        ]

        # build the stable wait list once, with known fixed indices
        if self.is_root:
            self._wait_list  = self.child_recv_reqs + [self.stop_recv_req]
            self._stop_idx   = len(self.children)
        else:
            self._wait_list  = self.child_recv_reqs + [self.parent_recv_req, self.stop_recv_req]
            self._parent_idx = len(self.children)
            self._stop_idx   = len(self.children) + 1

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _event_loop(self):
        self.timer.reset()
        with self.timer.measure("train"):
            while self.active_children > 0:
                with self.timer.measure("wait") as t:
                    idx = MPI.Request.Waitany(self._wait_list)

                if idx == MPI.UNDEFINED:
                    self.log("UNDEFINED INDEX (this should not happen)")
                    break

                if idx == self._stop_idx:
                    t["tag"] = "stop"
                    self._dispatch_stop()
                    break
                elif not self.is_root and idx == self._parent_idx:
                    t["tag"] = self.parent
                    self._dispatch_parent_push()
                else:
                    t["tag"] = self.children[idx]
                    self._dispatch_child_grad(idx)

                self._prune_send_queues()

    # ------------------------------------------------------------------
    # Dispatch handlers
    # ------------------------------------------------------------------

    def _dispatch_stop(self):
        self.eval_stopped = True
        self.log("Stop signal received, propagating to children")
        for i, child in enumerate(self.children):
            req = self.comm.Isend(self._stop_bufs[i], dest=child, tag=Msg.STOP)
            self.stop_reqs.append((req, self._stop_bufs[i]))

    def _dispatch_parent_push(self):
        self.model_version = int(self.parent_packet[0])
        self.flat_params[:] = self.parent_packet[1:]
        self.anchor_params[:] = self.flat_params

        if self.sync:
            with self.timer.measure("send_params") as ts:
                for child in self.children:
                    self.child_sq.send(self.parent_packet, dest=child, tag=Msg.PUSH_PARAMS)
                    ts["tag"] = child

        self.parent_recv_req = self.comm.Irecv(self.parent_packet, source=self.parent, tag=Msg.PUSH_PARAMS)
        self._wait_list[self._parent_idx] = self.parent_recv_req

    def _dispatch_child_grad(self, child_idx):
        src    = self.children[child_idx]
        packet = self.child_packets[child_idx]

        grad_version = int(packet[0])
        grads        = packet[1:]

        staleness = self.model_version - grad_version
        if staleness > 0:
            self.log(f"stale grad from rank {src}: computed on version {grad_version}, current {self.model_version}, staleness {staleness}")

        np.multiply(grads, self._lr_scale, out=self._update_buf)
        self.flat_params -= self._update_buf
        self.n_updates   += 1

        self.child_recv_reqs[child_idx] = self.comm.Irecv(
            self.child_packets[child_idx], source=src, tag=Msg.PUSH_GRADS
        )
        self._wait_list[child_idx] = self.child_recv_reqs[child_idx]

        self._apply_sync_or_async(src, grad_version)

    # ------------------------------------------------------------------
    # Update policy
    # ------------------------------------------------------------------

    def _apply_sync_or_async(self, src, grad_version):
        if self.sync:
            self.children_done.add(src)
            if len(self.children_done) < self.active_children:
                return

            # all children have reported — commit the round
            self._increment_version()

            with self.timer.measure("send_params") as ts:
                for child in self.children_done:
                    buf = self._claim_param_buf(self.child_sq)
                    self.child_sq.post(buf, dest=child, tag=Msg.PUSH_PARAMS)
                    ts["tag"] = child

            if not self.is_root:
                buf = self.parent_sq.claim()
                buf[0] = self.model_version
                np.subtract(self.anchor_params, self.flat_params, out=buf[1:])
                with self.timer.measure("send_params") as ts:
                    self.parent_sq.post(buf, dest=self.parent, tag=Msg.PUSH_GRADS)
                    ts["tag"] = self.parent

            self.children_done.clear()

            if self.is_root:
                self._maybe_push_eval()

        else:  # async
            self._increment_version()

            with self.timer.measure("send_params") as ts:
                buf = self._claim_param_buf(self.child_sq)
                self.child_sq.post(buf, dest=src, tag=Msg.PUSH_PARAMS)
                ts["tag"] = src

            if self.is_root:
                self._maybe_push_eval()
            else:
                if self.n_updates >= max(1, self.active_children // 2):
                    buf = self.parent_sq.claim()
                    buf[0] = self.model_version
                    np.subtract(self.anchor_params, self.flat_params, out=buf[1:])
                    with self.timer.measure("send_params") as ts:
                        self.parent_sq.post(buf, dest=self.parent, tag=Msg.PUSH_GRADS)
                        ts["tag"] = self.parent
                    self.anchor_params[:] = self.flat_params
                    self.n_updates = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _increment_version(self):
        self.model_version += 1
        assert self.model_version < 2**24, \
            f"model_version {self.model_version} exceeds float32 precision"

    def _claim_param_buf(self, sq):
        """Claim a pool buffer from sq and write the current versioned params into it."""
        buf = sq.claim()
        buf[0] = self.model_version
        buf[1:] = self.flat_params
        return buf

    def _maybe_push_eval(self):
        if self.eval_stopped:
            return
        if self.model_version % self.eval_interval != 0:
            return
        self.log(f"sending params to evaluator, version: {self.model_version}")
        with self.timer.measure("send_params") as ts:
            buf = self._claim_param_buf(self.child_sq)
            self.child_sq.post(buf, dest=self.evaluator, tag=Msg.EVAL)
            ts["tag"] = "eval"

    def _prune_send_queues(self):
        self.child_sq.prune()
        if not self.is_root:
            self.parent_sq.prune()

        gs = self.global_state
        gs["send_reqs_children"] = self.child_sq.pending_dests()
        gs["n_child_send_reqs"]  = len(self.child_sq)
        if not self.is_root:
            gs["n_parent_send_reqs"] = len(self.parent_sq)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self):
        role_name = "root ps" if self.is_root else "edge ps"
        self.log(f"{role_name} terminating...")

        self.log(f"waiting to send {len(self.stop_reqs)} stop signals...")
        for req, _ in self.stop_reqs:
            safe_wait_send_req(req, timeout=2.0)

        if self.is_root:
            for child in self.children:
                while self.comm.Iprobe(source=child, tag=Msg.PUSH_GRADS):
                    self.comm.Recv(MPI.IN_PLACE, source=child, tag=Msg.PUSH_GRADS)
        else:
            self.log("cancelling parent recv req")
            cancel_recv_req(self.parent_recv_req)

        self.log("cancelling child recv reqs")
        for req in self.child_recv_reqs:
            cancel_recv_req(req)

        if not self.is_root:
            self.log(f"cancelling {len(self.parent_sq)} parent send reqs")
            self.parent_sq.cancel_all()

        self.log(f"safe waiting {len(self.child_sq)} child send reqs")
        self.child_sq.drain(timeout=0.5)

        self.log(f"{role_name} done cleaning up")
