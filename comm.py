import time
import numpy as np
from collections import deque
from enum import IntEnum
from mpi4py import MPI


class Role(IntEnum):
    ROOT_PS = 1
    EDGE_PS = 2
    WORKER  = 3
    EVAL    = 4


class Msg(IntEnum):
    PUSH_PARAMS = 1
    PUSH_GRADS  = 2
    STOP        = 3
    EVAL        = 4


def cleanup_mpi_req(req):
    if req is not None and req != MPI.REQUEST_NULL:
        req.Cancel()
        try:
            MPI.Request.Wait(req)
        except MPI.Exception:
            pass


def cancel_recv_req(req):
    if req is not None and req != MPI.REQUEST_NULL:
        req.Cancel()
        try:
            MPI.Request.Wait(req)
        except MPI.Exception:
            pass


def safe_wait_send_req(req, timeout=1.0):
    if req is not None and req != MPI.REQUEST_NULL:
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout:
            if req.Test():
                return
            time.sleep(0.01)
        req.Cancel()


class SendQueue:
    """
    Tracks in-flight non-blocking MPI sends and manages a fixed pool of
    reusable send buffers.

    Instead of allocating a fresh buffer on every send(), buffers are recycled
    from a pool: send() claims a free buffer from the pool and posts the Isend;
    prune() returns completed buffers back to the pool. Back-pressure spins
    until a buffer is available if the pool is exhausted.

    buf_shape: numpy shape tuple for buffer pre-allocation. If None, the pool
    is initialized lazily from the shape of the first buffer passed to send().
    """

    def __init__(self, comm, max_inflight, buf_shape=None):
        self._comm         = comm
        self._max_inflight = max_inflight
        self._inflight     = []           # list of (req, buf, dst)
        self._pool         = deque()      # free buffers available for reuse
        if buf_shape is not None:
            self._init_pool(buf_shape)

    def _init_pool(self, shape):
        for _ in range(self._max_inflight):
            self._pool.append(np.empty(shape, dtype=np.float32))

    def send(self, buf, dest, tag):
        # lazily initialize pool on first send if buf_shape wasn't given
        if not self._pool and not self._inflight:
            self._init_pool(buf.shape)

        # spin until a buffer is free
        while not self._pool:
            self.prune()

        owned = self._pool.popleft()
        np.copyto(owned, buf)
        req = self._comm.Isend(owned, dest=dest, tag=tag)
        self._inflight.append((req, owned, dest))

    def claim(self):
        """Return a free pool buffer for the caller to fill directly.

        Spins (pruning completed sends) until a buffer is available.
        Must be followed by exactly one call to post() with the returned buffer.
        """
        if not self._pool and not self._inflight:
            raise RuntimeError("SendQueue.claim() called before pool was initialized; pass buf_shape to __init__")
        while not self._pool:
            self.prune()
        return self._pool.popleft()

    def post(self, buf, dest, tag):
        """Post an Isend for a buffer previously obtained via claim()."""
        req = self._comm.Isend(buf, dest=dest, tag=tag)
        self._inflight.append((req, buf, dest))

    def prune(self):
        still_flying = []
        for req, buf, dst in self._inflight:
            if req.Test():
                self._pool.append(buf)   # return buffer to pool
            else:
                still_flying.append((req, buf, dst))
        self._inflight = still_flying

    def pending_dests(self):
        return [d for _, _, d in self._inflight]

    def drain(self, timeout=0.5):
        for req, buf, _ in self._inflight:
            safe_wait_send_req(req, timeout=timeout)
            self._pool.append(buf)
        self._inflight.clear()

    def cancel_all(self):
        for req, buf, _ in self._inflight:
            cleanup_mpi_req(req)
            self._pool.append(buf)
        self._inflight.clear()

    def __len__(self):
        return len(self._inflight)
