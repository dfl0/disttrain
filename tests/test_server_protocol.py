"""Synchronous round arithmetic and admission with real tensors and sockets."""
from contextlib import contextmanager, ExitStack
import unittest

import numpy as np
import torch
import zmq

from protocol import (Msg, Role, ParamSerializer, encode_packet, decode_packet,
                      encode_assignment, decode_assignment, encode_version)
from ps import ParameterServer
from transport import open_control_socket, send_message, receive_message, open_role_sockets


@contextmanager
def server(children=('w0', 'w1'), ranks=None, edge=False):
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.RCVTIMEO, 2000)
    router.bind('inproc://server')
    peers = {}
    parent_peer = dealer = None
    try:
        for name in children:
            sock = context.socket(zmq.DEALER)
            sock.setsockopt(zmq.IDENTITY, name.encode())
            sock.setsockopt(zmq.RCVTIMEO, 2000)
            sock.connect('inproc://server')
            peers[name] = sock
            send_message(sock, Msg.REGISTER, name.encode())
        if edge:
            parent_peer = context.socket(zmq.DEALER)
            parent_peer.setsockopt(zmq.RCVTIMEO, 2000)
            parent_peer.bind('inproc://parent')
            dealer = context.socket(zmq.DEALER)
            dealer.setsockopt(zmq.RCVTIMEO, 2000)
            dealer.connect('inproc://parent')
            send_message(parent_peer, Msg.ASSIGN, encode_assignment(-1))
            send_message(parent_peer, Msg.PUSH_PARAMS, encode_packet(0, np.zeros(1, dtype=np.float32)))
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        ps = ParameterServer(model, children, True, router, ParamSerializer(model), lambda module: None,
                             child_ranks=ranks or {name: i for i, name in enumerate(children)},
                             parent='root' if edge else None, dealer_sock=dealer,
                             node_id='edge' if edge else 'root', lr=0.1)
        ps._init_params()
        if edge:
            assert receive_message(parent_peer)[1:] == (bytes([Msg.REGISTER]), b'edge')
        yield ps, peers, parent_peer
    finally:
        context.destroy(linger=0)


def consume_initial(test, ps, peers):
    for name, peer in peers.items():
        _, kind, payload = receive_message(peer)
        test.assertEqual(kind, bytes([Msg.ASSIGN]))
        test.assertEqual(decode_assignment(payload), ps.child_ranks[name])
        _, kind, payload = receive_message(peer)
        test.assertEqual(kind, bytes([Msg.PUSH_PARAMS]))
        version, _, values = decode_packet(payload, 1)
        test.assertEqual(version, 0)
        np.testing.assert_array_equal(values, [0])


class ServerProtocolTests(unittest.TestCase):
    def test_global_assignments_and_weighted_round(self):
        with server(ranks={'w0': 3, 'w1': 8}) as (ps, peers, _):
            consume_initial(self, ps, peers)
            ps._accept_contribution('w0', encode_packet(0, np.array([20], dtype=np.float32), 10))
            np.testing.assert_array_equal(ps.flat_params, [0])
            self.assertEqual(peers['w0'].poll(0), 0)
            ps._accept_contribution('w1', encode_packet(0, np.array([120], dtype=np.float32), 30))
            np.testing.assert_allclose(ps.flat_params, [-0.35])
            self.assertEqual((ps.version, ps.sample_count, ps.received), (1, 0, set()))
            for peer in peers.values():
                _, kind, payload = receive_message(peer)
                self.assertEqual(kind, bytes([Msg.PUSH_PARAMS]))
                self.assertEqual(decode_packet(payload, 1)[0], 1)

    def test_flat_matches_unequal_edge_fanout_and_edges_wait(self):
        contributions = [('a', 2., 1), ('b', 12., 3), ('c', 10., 2)]
        with ExitStack() as stack:
            flat, flat_peers, _ = stack.enter_context(server(('a', 'b', 'c')))
            root, root_peers, _ = stack.enter_context(server(('e0', 'e1'), ranks={'e0': -1, 'e1': -1}))
            e0, peers0, parent0 = stack.enter_context(server(('a',), edge=True))
            e1, peers1, parent1 = stack.enter_context(server(('b', 'c'), edge=True))
            for ps, peers in [(flat, flat_peers), (root, root_peers), (e0, peers0), (e1, peers1)]:
                consume_initial(self, ps, peers)
            for name, total, count in contributions:
                packet = encode_packet(0, np.array([total], dtype=np.float32), count)
                flat._accept_contribution(name, packet)
                (e0 if name == 'a' else e1)._accept_contribution(name, packet)
            for name, edge, peers, parent in [('e0', e0, peers0, parent0), ('e1', e1, peers1, parent1)]:
                self.assertTrue(edge.waiting_for_parent)
                np.testing.assert_array_equal(edge.flat_params, [0])
                for peer in peers.values():
                    self.assertEqual(peer.poll(0), 0)
                _, kind, payload = receive_message(parent)
                self.assertEqual(kind, bytes([Msg.PUSH_GRADS]))
                root._accept_contribution(name, payload)
            np.testing.assert_allclose(root.flat_params, flat.flat_params)
            np.testing.assert_allclose(root.flat_params, [-0.4])
            for name, edge, peers in [('e0', e0, peers0), ('e1', e1, peers1)]:
                _, _, payload = receive_message(root_peers[name])
                edge._dispatch_parent_push(payload)
                self.assertFalse(edge.waiting_for_parent)
                self.assertEqual(edge.sample_count, 0)
                np.testing.assert_allclose(edge.flat_params, root.flat_params)
                with self.assertRaises(ValueError):
                    edge._dispatch_parent_push(payload)  # duplicate parent response
                for peer in peers.values():
                    self.assertEqual(decode_packet(receive_message(peer)[2], 1)[0], 1)
                    self.assertEqual(peer.poll(0), 0)

            # All workers finish at the same committed version, through both edges.
            for name, edge, peers, parent in [('e0', e0, peers0, parent0), ('e1', e1, peers1, parent1)]:
                for child in peers:
                    edge._accept_done(child, encode_version(1))
                _, kind, payload = receive_message(parent)
                self.assertEqual(kind, bytes([Msg.DONE]))
                root._accept_done(name, payload)
            self.assertTrue(root.stopped)
            for name, edge, peers in [('e0', e0, peers0), ('e1', e1, peers1)]:
                self.assertEqual(receive_message(root_peers[name])[1:], (bytes([Msg.STOP]), b''))
                edge._dispatch_stop()
                for peer in peers.values():
                    self.assertEqual(receive_message(peer)[1:], (bytes([Msg.STOP]), b''))

    def test_rejection_before_round_mutation(self):
        with server() as (ps, peers, _):
            consume_initial(self, ps, peers)
            valid = encode_packet(0, np.ones(1, dtype=np.float32), 1)
            ps._accept_contribution('w0', valid)
            for child, packet in [('w0', valid), ('unknown', valid),
                                  ('w1', encode_packet(1, np.ones(1, dtype=np.float32), 1)),
                                  ('w1', encode_packet(0, np.ones(2, dtype=np.float32), 1)),
                                  ('w1', encode_packet(0, np.ones(1, dtype=np.float32))),
                                  ('w1', valid[:-1])]:
                with self.subTest(child=child, payload=packet), self.assertRaises(ValueError):
                    ps._accept_contribution(child, packet)
                self.assertEqual((ps.sample_count, ps.received), (1, {'w0'}))
                np.testing.assert_array_equal(ps.gradient_sum, [1])
                np.testing.assert_array_equal(ps.flat_params, [0])

    def test_completion_requires_all_children_and_propagates_stop(self):
        with server() as (ps, peers, _):
            consume_initial(self, ps, peers)
            ps._accept_done('w0', encode_version(0))
            self.assertFalse(ps.stopped)
            with self.assertRaises(ValueError):
                ps._accept_contribution('w1', encode_packet(0, np.ones(1, dtype=np.float32), 1))
            ps._accept_done('w1', encode_version(0))
            self.assertTrue(ps.stopped)
            for peer in peers.values():
                self.assertEqual(receive_message(peer)[1:], (bytes([Msg.STOP]), b''))

    def test_integer_codec_and_reject_malformed_payloads(self):
        values = np.array([1, 2], dtype=np.float32)
        payload = encode_packet(2**40, values, 7)
        values[:] = 99
        version, count, decoded = decode_packet(payload, 2, gradients=True)
        self.assertEqual((version, count), (2**40, 7))
        np.testing.assert_array_equal(decoded, [1, 2])
        for bad in (payload[:-1], payload + b'x', b'BAD!' + payload[4:]):
            with self.assertRaises(ValueError):
                decode_packet(bad, 2, gradients=True)
        bad = payload[:-4] + np.array([np.nan], dtype='<f4').tobytes()
        with self.assertRaises(ValueError):
            decode_packet(bad, 2, gradients=True)
    def test_round_limit_stops_without_releasing_another_round(self):
        with server() as (ps, peers, _):
            consume_initial(self, ps, peers)
            ps.max_rounds = 1
            for name in peers:
                ps._accept_contribution(name, encode_packet(0, np.ones(1, dtype=np.float32), 1))
            self.assertEqual(ps.version, 1)
            self.assertTrue(ps.stopped)
            np.testing.assert_allclose(ps.flat_params, [-0.1])
            for peer in peers.values():
                self.assertEqual(receive_message(peer)[1:], (bytes([Msg.STOP]), b''))
                self.assertFalse(peer.poll(0))

class SocketWiringTests(unittest.TestCase):
    def test_partial_socket_setup_releases_only_its_own_sockets(self):
        context = zmq.Context()
        blocker = context.socket(zmq.PULL)
        blocker.bind('inproc://occupied')
        try:
            with self.assertRaises(zmq.ZMQError):
                open_role_sockets(context, Role.ROOT_PS,
                                  dict(bind_endpoint='inproc://training'),
                                  dict(eval_endpoint='invalid://eval'), 'root')
            # The first listener must be released after the second bind fails.
            replacement = context.socket(zmq.ROUTER)
            replacement.bind('inproc://training')
            self.assertFalse(blocker.closed)
        finally:
            context.destroy(linger=0)

    def test_roles_connect_to_configured_peers(self):
        context = zmq.Context()
        topology = dict(eval_endpoint='inproc://eval', stop_endpoint='inproc://stop')
        sockets = []
        try:
            for role, node_id, cfg in [
                (Role.ROOT_PS, 'root', dict(bind_endpoint='inproc://root', stop_bind_endpoint='inproc://stop')),
                (Role.EDGE_PS, 'edge', dict(bind_endpoint='inproc://edge', parent_endpoint='inproc://root')),
                (Role.WORKER, 'worker', dict(parent_endpoint='inproc://edge')),
                (Role.EVAL, 'eval', dict(bind_endpoint='inproc://eval')),
            ]:
                sockets.append(open_role_sockets(context, role, cfg, topology, node_id))
            root, edge, worker, evaluator = sockets
            for sender, receiver, identity in [(edge.dealer, root.router, b'edge'),
                                               (worker.dealer, edge.router, b'worker')]:
                receiver.setsockopt(zmq.RCVTIMEO, 2000)
                send_message(sender, Msg.REGISTER, identity)
                self.assertEqual(receive_message(receiver, routed=True),
                                 (identity, bytes([Msg.REGISTER]), identity))
            for sender, receiver, message in [(root.evaluator, evaluator.pull, Msg.EVAL)]:
                receiver.setsockopt(zmq.RCVTIMEO, 2000)
                send_message(sender, message)
                self.assertEqual(receive_message(receiver), (None, bytes([message]), b''))
            control = open_control_socket(context, Role.ROOT_PS,
                                          dict(stop_bind_endpoint='inproc://stop'), topology, 'root')
            for role, node_id in ((Role.WORKER, 'worker'), (Role.EDGE_PS, 'edge'), (Role.EVAL, 'eval')):
                peer = open_control_socket(context, role, {}, topology, node_id)
                peer.send_multipart([b'READY', b''])
                self.assertTrue(control.poll(2000))
                self.assertEqual(control.recv_multipart(), [node_id.encode(), b'READY', b''])
                control.send_multipart([node_id.encode(), b'START', b''])
                self.assertTrue(peer.poll(2000))
                self.assertEqual(peer.recv_multipart(), [b'START', b''])
        finally:
            context.destroy(linger=0)
