"""Cooperative run control over a root ROUTER and one DEALER per participant.

No retries, heartbeats, threads, or recovery. Training sockets remain separate.
"""
import json
import math
import time

import zmq


class RunStopped(Exception):
    """Unwind a role to its runtime-owned shutdown handshake."""

    def __init__(self, error=''):
        super().__init__(error)
        self.error = error


def failure_report(reason, *, interrupted=False):
    return json.dumps(dict(reason=reason, interrupted=interrupted))


def merge_reports(first, second):
    """Keep a real failure over an interruption, regardless of arrival order."""
    if not first:
        return second
    if not second:
        return first
    return second if json.loads(first)['interrupted'] and not json.loads(second)['interrupted'] else first


class RunFailed(RuntimeError):
    """A non-success run outcome, distinct from failure of the shutdown handshake."""
    def __init__(self, report):
        outcome = json.loads(report)
        self.interrupted = outcome['interrupted']
        super().__init__(outcome['reason'])



class Lifecycle:
    def __init__(self, socket, node_id, root_id, participants, evaluator_id, timeouts):
        self.socket, self.node_id = socket, node_id
        self.is_root = node_id == root_id
        self.peers = set(participants) - {root_id}
        self.evaluator_id = evaluator_id
        self.timeouts = timeouts
        self.deadline = time.monotonic() + timeouts['startup_s']
        self.work_report = None
        self.work_reports = {}
        self.last_evaluated = -1
        self.stop_received = False
        self.error = ''
        self.shutdown_deadline = None
        self.clock_recorder = None

    @staticmethod
    def _milliseconds(deadline):
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('communication/progress deadline expired')
        return max(1, math.ceil(remaining * 1000))

    def _send(self, kind, payload='', *, peer=None, deadline=None):
        frames = [kind.encode(), payload.encode()]
        if self.is_root:
            frames.insert(0, peer.encode())
        while True:
            self._milliseconds(deadline)
            try:
                self.socket.send_multipart(frames, flags=zmq.DONTWAIT)
                return
            except zmq.Again:
                if not self.socket.poll(self._milliseconds(deadline), zmq.POLLOUT):
                    raise TimeoutError('control send deadline expired')

    def _receive(self):
        frames = self.socket.recv_multipart()
        if len(frames) != (3 if self.is_root else 2):
            raise ValueError('invalid control frame count')
        peer = frames.pop(0).decode() if self.is_root else None
        if self.is_root and peer not in self.peers:
            raise ValueError(f'unknown control participant {peer!r}')
        kind, payload = (frame.decode() for frame in frames)
        return peer, kind, payload

    def _clock_reply(self, kind, payload, deadline):
        if self.is_root or kind != 'CLOCK':
            return False
        received = time.monotonic_ns()
        sent = time.monotonic_ns()
        self._send('CLOCK_REPLY', json.dumps([int(payload), received, sent]), deadline=deadline)
        return True

    def _calibrate(self, phase, deadline):
        """Record the lowest-delay clock sample within the phase deadline."""
        if self.clock_recorder is None:
            return
        budget = min(deadline, time.monotonic() + 1.0)
        for target in sorted(self.peers):
            best = None
            try:
                for _ in range(5):
                    first = time.monotonic_ns()
                    self._send('CLOCK', str(first), peer=target, deadline=budget)
                    while True:
                        if not self.socket.poll(self._milliseconds(budget)):
                            raise TimeoutError('clock calibration expired')
                        peer, kind, payload = self._receive()
                        last = time.monotonic_ns()
                        if kind == 'FAIL':
                            raise RunStopped(payload)
                        if kind != 'CLOCK_REPLY':
                            raise ValueError(f'unexpected calibration message {kind}')
                        echoed, received, sent = json.loads(payload)
                        if peer == target and echoed == first:
                            break  # ignore replies to expired probes
                    uncertainty = (last - first - (sent - received)) / 2
                    if sent < received or uncertainty < 0:
                        continue
                    sample = ((received + sent) // 2, (first + last) // 2, math.ceil(uncertainty))
                    if best is None or sample[2] < best[2]:
                        best = sample
            except (TimeoutError, zmq.ZMQError):
                return  # keep any completed samples
            finally:
                if best is not None:
                    self.clock_recorder.record('clock_calibration', target, phase, *best)

    def _handle(self):
        peer, kind, payload = self._receive()
        if self.is_root:
            if kind == 'CLOCK_REPLY':
                return  # ignore replies to expired probes
            if kind == 'FAIL':
                self.shutdown_deadline = time.monotonic() + self.timeouts['shutdown_s']
                raise RunStopped(payload)
            if kind == 'STOP_REQUEST' and peer == self.evaluator_id and not payload:
                self.shutdown_deadline = time.monotonic() + self.timeouts['shutdown_s']
                raise RunStopped()
            if kind == 'EVALUATED' and peer == self.evaluator_id:
                version = int(payload)
                if version < self.last_evaluated or version < 0:
                    raise ValueError('invalid evaluated version')
                self.last_evaluated = version
                return
        elif kind == 'STOP':
            self.stop_received, self.error = True, payload
            self.shutdown_deadline = time.monotonic() + self.timeouts['shutdown_s']
            raise RunStopped(payload)
        raise ValueError(f'unexpected control message {kind}')

    def ready(self):
        """Barrier after model/data setup; the deadline also covers registration."""
        if self.is_root:
            ready = set()
            while ready != self.peers:
                if not self.socket.poll(self._milliseconds(self.deadline)):
                    raise TimeoutError('startup: participants did not become ready')
                peer, kind, payload = self._receive()
                if kind == 'FAIL':
                    raise RunStopped(payload)
                if kind != 'READY' or payload or peer in ready:
                    raise ValueError('expected unique READY from each participant')
                ready.add(peer)
            self._calibrate('startup', self.deadline)
            for peer in self.peers:
                self._send('START', peer=peer, deadline=self.deadline)
        else:
            self._send('READY', deadline=self.deadline)
            while True:
                if not self.socket.poll(self._milliseconds(self.deadline)):
                    raise TimeoutError('startup: root did not release participants')
                _, kind, payload = self._receive()
                if self._clock_reply(kind, payload, self.deadline):
                    continue
                if kind == 'STOP':
                    self.stop_received, self.error = True, payload
                    raise RunStopped(payload)
                if kind != 'START' or payload:
                    raise ValueError('expected START from root')
                break

    def begin_round(self):
        seconds = self.timeouts['training_s']
        self.deadline = None if seconds is None else time.monotonic() + seconds

    def check(self):
        if getattr(self, "health_check", None):
            self.health_check()
        # Control takes priority even when the training socket is already ready.
        while self.socket.poll(0):
            self._handle()
        self._milliseconds(self.deadline)

    def wait(self, sockets, *, event=zmq.POLLIN, deadline=None):
        """Wait for readiness, not a periodic check. Bound by the current phase."""
        if self.deadline is not None:
            deadline = self.deadline if deadline is None else min(deadline, self.deadline)
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        for socket in sockets:
            poller.register(socket, event)
        while True:
            ready = dict(poller.poll(self._milliseconds(deadline)))
            if self.socket in ready:
                self._handle()
                del ready[self.socket]
            if ready:
                return ready
            if not ready and deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('communication/progress deadline expired')

    def receive(self, socket, *, routed=False):
        from transport import receive_message
        self.wait([socket])
        return receive_message(socket, routed=routed)

    def send(self, socket, message, payload=b'', *, identity=None):
        frames = [bytes([message]), payload]
        if identity is not None:
            frames.insert(0, identity)
        while True:
            self.check()
            try:
                socket.send_multipart(frames, flags=zmq.DONTWAIT)
                return
            except zmq.Again:
                self.wait([socket], event=zmq.POLLOUT)

    def evaluated(self, version):
        self._send('EVALUATED', str(version), deadline=self.deadline)

    def request_stop(self):
        self.shutdown_deadline = time.monotonic() + self.timeouts['shutdown_s']
        self.deadline = self.shutdown_deadline
        self._send('STOP_REQUEST', deadline=self.deadline)
        self.wait([])

    def wait_evaluation(self, version):
        while self.last_evaluated < version:
            self.wait_control()

    def wait_control(self):
        if not self.socket.poll(self._milliseconds(self.deadline)):
            raise TimeoutError('control deadline expired')
        self._handle()

    def finish(self, error=''):
        """ACK means the role has unwound; FINISH confirms root received the ACK.

        All waits share one shutdown budget. Missing peers make shutdown a failure.
        """
        if self.shutdown_deadline is None:
            self.shutdown_deadline = time.monotonic() + self.timeouts['shutdown_s']
        deadline = self.shutdown_deadline
        error = merge_reports(error, self.error)
        if not self.is_root:
            if error and not self.stop_received:
                self._send('FAIL', error, deadline=deadline)
            while not self.stop_received:
                if not self.socket.poll(self._milliseconds(deadline)):
                    raise TimeoutError('shutdown: no root STOP')
                _, kind, payload = self._receive()
                if kind == 'STOP':
                    self.stop_received, self.error = True, payload
                elif self._clock_reply(kind, payload, deadline):
                    continue
                elif kind != 'START':
                    raise ValueError('shutdown: expected STOP')
            error = merge_reports(error, self.error)
            self._send('ACK', json.dumps(dict(error=error, work=self.work_report)), deadline=deadline)
            while True:
                if not self.socket.poll(self._milliseconds(deadline)):
                    raise TimeoutError('shutdown: root did not confirm acknowledgement')
                _, kind, payload = self._receive()
                if not self._clock_reply(kind, payload, deadline):
                    break
            if kind == 'INCOMPLETE':
                raise TimeoutError(payload)
            if kind != 'FINISH':
                raise ValueError('shutdown: expected FINISH')
            error = merge_reports(error, payload)
        else:
            shutdown_error = ''
            acknowledged = set()
            notified = set()
            for peer in self.peers:
                try:
                    self._send('STOP', error, peer=peer, deadline=deadline)
                    notified.add(peer)
                except zmq.ZMQError as exc:
                    if exc.errno != zmq.EHOSTUNREACH:
                        raise
                    # A late READY/FAIL supplies a route; otherwise deadline fails.
            try:
                while acknowledged != self.peers:
                    if not self.socket.poll(self._milliseconds(deadline)):
                        raise TimeoutError('shutdown acknowledgements missing')
                    peer, kind, payload = self._receive()
                    if kind == 'ACK':
                        acknowledged.add(peer)
                        ack = json.loads(payload)
                        error = merge_reports(error, ack['error'])
                        work = ack['work']
                        if work is not None and (not isinstance(work, dict)
                                or set(work) != {'contributions', 'samples', 'partial_samples'}
                                or any(type(value) is not int or value < 0 for value in work.values())):
                            raise ValueError('invalid shutdown work report')
                        self.work_reports[peer] = work
                    elif kind in ('READY', 'FAIL'):
                        if kind == 'FAIL':
                            error = merge_reports(error, payload)
                        if peer not in notified:
                            self._send('STOP', error, peer=peer, deadline=deadline)
                            notified.add(peer)
                    elif kind not in ('EVALUATED', 'STOP_REQUEST', 'CLOCK_REPLY'):
                        raise ValueError('unexpected shutdown control message')
            except TimeoutError:
                shutdown_error = f'missing acknowledgements from {sorted(self.peers - acknowledged)}'
            if not shutdown_error and not error:
                self._calibrate('shutdown', deadline)
            # Nonblocking final confirmations; bounded linger below permits flush.
            for peer in acknowledged:
                try:
                    self.socket.send_multipart([peer.encode(),
                                                b'INCOMPLETE' if shutdown_error else b'FINISH',
                                                (shutdown_error or error).encode()], flags=zmq.DONTWAIT)
                except zmq.ZMQError:
                    # preserve the missing-ACK error if a peer has disconnected
                    if not shutdown_error:
                        raise
        # A received FINISH proves participant ACK delivery. Root gets a bounded
        # flush budget for FINISH, not the old unconditional two-second drain.
        self.socket.close(linger=max(0, math.ceil((deadline - time.monotonic()) * 1000)))
        if self.is_root and shutdown_error:
            raise TimeoutError(shutdown_error)
        if error:
            raise RunFailed(error)
