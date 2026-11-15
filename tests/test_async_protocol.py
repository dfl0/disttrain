"""Test async updates and edge aggregation using local sockets."""
import unittest
import numpy as np
from test_server_protocol import server
from protocol import (Msg, encode_async_gradients, decode_async_gradients,
                      encode_async_params, decode_async_params, encode_version)
from transport import receive_message


def packet(rank, contribution=1, source=0, samples=2, gradient=4):
    return encode_async_gradients(contribution, [(rank, contribution, source, samples)],
                                  np.array([gradient], dtype=np.float32))


def initial(peers):
    for peer in peers.values():
        receive_message(peer)  # assignment
        assert decode_async_params(receive_message(peer)[2], 1)[:2] == (0, 0)


class AsyncProtocolTests(unittest.TestCase):
    def test_fixed_normalization_staleness_and_shrinking_membership(self):
        with server(asynchronous=True) as (root, peers, _):
            initial(peers)
            root._accept_contribution('w0', packet(0))
            self.assertEqual(decode_async_params(receive_message(peers['w0'])[2], 1)[:2], (1, 1))
            self.assertFalse(peers['w1'].poll(0), 'no unsolicited update')
            root._accept_done('w0', encode_version(1))
            root._accept_contribution('w1', packet(1, samples=1, gradient=2))
            np.testing.assert_allclose(root.flat_params, [-0.075])  # use the fixed reference of 8 samples
            self.assertEqual((root.total_samples, root.weighted_staleness, root.max_staleness), (3, 1, 1))
            receive_message(peers['w1'])
            root._accept_contribution('w1', packet(1, contribution=2, source=2, gradient=8))
            np.testing.assert_allclose(root.flat_params, [-0.175])  # reference stays 8
            receive_message(peers['w1'])
            root._accept_done('w1', encode_version(2))
            self.assertTrue(root.stopped)

    def test_edge_one_inflight_batches_pending_and_replies_only_to_members(self):
        with server(children=('a', 'b', 'c'), edge=True, asynchronous=True) as (edge, peers, parent):
            initial(peers)
            edge._accept_contribution('a', packet(0))
            first = decode_async_gradients(receive_message(parent)[2], 1)
            self.assertEqual(first[:2], (1, [(0, 1, 0, 2)]))
            edge._accept_contribution('b', packet(1, gradient=6))
            edge._accept_contribution('c', packet(2, gradient=8))
            self.assertFalse(parent.poll(0))
            with self.assertRaises(ValueError):
                edge._accept_contribution('a', packet(0, contribution=2))
            with self.assertRaises(ValueError):
                edge._dispatch_parent_push(encode_async_params(2, 1, np.array([0], dtype=np.float32)))
            edge._dispatch_parent_push(encode_async_params(1, 1, np.array([1], dtype=np.float32)))
            self.assertEqual(decode_async_params(receive_message(peers['a'])[2], 1)[:2], (1, 1))
            self.assertFalse(peers['b'].poll(0))
            request, records, gradients = decode_async_gradients(receive_message(parent)[2], 1)
            self.assertEqual(request, 2)
            self.assertEqual(records, [(1, 1, 0, 2), (2, 1, 0, 2)])
            np.testing.assert_array_equal(gradients, [14])
            edge._accept_done('a', encode_version(1))
            edge._dispatch_parent_push(encode_async_params(2, 4, np.array([2], dtype=np.float32)))
            for name in ('b', 'c'):
                self.assertEqual(decode_async_params(receive_message(peers[name])[2], 1)[:2], (1, 4))
                edge._accept_done(name, encode_version(1))
            self.assertEqual(receive_message(parent)[1:], (bytes([Msg.ASYNC_DONE]), encode_version(2)))

    def test_invalid_admission_is_atomic_and_mixed_sources_are_preserved(self):
        with server(children=('e',), ranks={'e': -1}, asynchronous=True, groups={'e': {0, 1}}) as (root, peers, _):
            initial(peers)
            root._accept_contribution('e', packet(0))
            receive_message(peers['e'])
            bad_records = [[(0, 1, 0, 2)], [(0, 2, 2, 2)], [(2, 1, 0, 2)], [(0, 2, 0, 2)]]
            for records in bad_records:
                with self.assertRaises(ValueError):
                    root._accept_contribution('e', encode_async_gradients(2, records, np.array([1], dtype=np.float32)))
                self.assertEqual(root.version, 1)
                self.assertEqual(root.requests['e'], 1)
            records = [(0, 2, 1, 2), (1, 1, 0, 1)]
            root._accept_contribution('e', encode_async_gradients(2, records, np.array([6], dtype=np.float32)))
            self.assertEqual(root.weighted_staleness, 1)
            np.testing.assert_allclose(root.flat_params, [-0.125])

    def test_codec_rejects_truncation_duplicate_records_and_nonfinite_values(self):
        valid = packet(0)
        for payload in (valid[:-1], valid + b'x', b'', b'xxxx' + valid[4:]):
            with self.assertRaises(ValueError):
                decode_async_gradients(payload, 1)
        for records in ([], [(0, 1, 0, 0)], [(0, 1, 0, 1)] * 2):
            with self.assertRaises(ValueError):
                encode_async_gradients(1, records, np.ones(1, dtype=np.float32))
        with self.assertRaises(ValueError):
            packet(0, gradient=float('nan'))
