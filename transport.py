"""ZeroMQ message framing, role socket setup, and existing cleanup policy.

These helpers do not provide backend interchangeability or reliable shutdown.
"""
import time
from dataclasses import dataclass

import zmq


def _open_socket(context, pattern, endpoint, *, bind=False, identity=None):
    sock = context.socket(pattern)
    try:
        if identity is not None:
            sock.setsockopt(zmq.IDENTITY, identity)
        if bind:
            sock.bind(endpoint)
        else:
            sock.connect(endpoint)
        return sock
    except BaseException:
        sock.close(0)
        raise


def send_message(sock, message, payload=b'', *, identity=None):
    frames = [bytes([message]), payload]
    if identity is not None:
        frames.insert(0, identity)
    sock.send_multipart(frames)


def receive_message(sock, *, routed=False):
    frames = sock.recv_multipart()
    from protocol import Msg
    offset = int(routed)
    if len(frames) != 2 + offset or len(frames[offset]) != 1:
        raise ValueError('invalid message frame count or type frame')
    identity = frames[0] if routed else None
    message, payload = frames[offset:]
    try:
        kind = Msg(message[0])
    except ValueError as error:
        raise ValueError('unknown message type') from error
    if kind == Msg.STOP and payload:
        raise ValueError('STOP must have an empty payload')
    return identity, message, payload


def close_sockets(*sockets):
    for sock in filter(None, sockets):
        sock.setsockopt(zmq.LINGER, 0)
        sock.close()


def drain_socket(sock, seconds=2.0):
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        remaining_ms = int((deadline - time.perf_counter()) * 1000)
        if not poller.poll(timeout=max(0, remaining_ms)):
            break
        sock.recv_multipart()


@dataclass
class RuntimeSockets:
    router: object = None
    dealer: object = None
    stop: object = None
    evaluator: object = None
    pull: object = None
    push: object = None


def open_role_sockets(context, role, cfg, topology, node_id):
    """Preserve role-specific socket creation order and existing endpoint policy."""
    from protocol import Role
    sockets = RuntimeSockets()
    try:
        if role in (Role.ROOT_PS, Role.EDGE_PS):
            sockets.router = _open_socket(context, zmq.ROUTER, cfg['bind_endpoint'], bind=True)
        if role == Role.ROOT_PS:
            sockets.evaluator = _open_socket(context, zmq.PUSH, topology['eval_endpoint'])
        elif role in (Role.EDGE_PS, Role.WORKER):
            sockets.dealer = _open_socket(context, zmq.DEALER, cfg['parent_endpoint'], identity=node_id.encode())
        elif role == Role.EVAL:
            sockets.pull = _open_socket(context, zmq.PULL, cfg['bind_endpoint'], bind=True)
        return sockets
    except BaseException:
        close_sockets(*vars(sockets).values())
        raise


def open_control_socket(context, role, cfg, topology, node_id):
    """Open before role/model setup so subsequent startup errors can be reported."""
    from protocol import Role
    if role == Role.ROOT_PS:
        sock = _open_socket(context, zmq.ROUTER, cfg['stop_bind_endpoint'], bind=True)
        sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
        return sock
    return _open_socket(context, zmq.DEALER, topology['stop_endpoint'], identity=node_id.encode())
