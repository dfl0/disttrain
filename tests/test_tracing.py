"""Test span recording and trace export."""
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest

from export_trace import build_trace
from metrics import MetricsRecorder, trace_span


class TracingTests(unittest.TestCase):
    def test_disabled_and_capture_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            disabled = MetricsRecorder(Path(tmp) / 'off')
            with trace_span(disabled, 'not_recorded'):
                pass
            disabled.close()
            self.assertFalse((disabled.directory / 'spans.csv').exists())
            recorder = MetricsRecorder(Path(tmp) / 'on', trace_duration_s=.01)
            start = recorder.begin_span()
            time.sleep(.02)
            self.assertIsNone(recorder.begin_span())
            # keep the full span if it starts within the capture window
            recorder.end_span(start, 'long_operation')
            recorder.close()
            with (recorder.directory / 'spans.csv').open() as file:
                row = next(csv.DictReader(file))
            self.assertGreater(int(row['end_ns']) - int(row['start_ns']), 10_000_000)

    def test_interrupted_nested_spans_and_shared_clock_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            nodes = {}
            for node in ('edge', 'worker'):
                recorder = MetricsRecorder(directory / f'{node}.metrics', trace_duration_s=10)
                exchange = recorder.begin_span()
                with self.assertRaisesRegex(ValueError, 'stop'):
                    with trace_span(recorder, 'handle_contribution', peer='worker'):
                        with trace_span(recorder, 'compute', version=2):
                            pass
                        raise ValueError('stop')
                recorder.end_span(exchange, 'upstream_exchange', track='upstream', request=3)
                recorder.close()
                nodes[node] = dict(role=node, metrics=f'{node}.metrics', metrics_status='complete')
            summary = directory / 'summary.json'
            summary.write_text(json.dumps(dict(mode='local', nodes=nodes)))
            trace = build_trace(directory)
            events = [event for event in trace['traceEvents'] if event['ph'] == 'X']
            self.assertEqual(len(events), 6)
            self.assertEqual({event['tid'] for event in events}, {0, 1})
            self.assertEqual(events[0]['ts'], 0)
            self.assertGreater(min(e['ts'] for e in events if e['pid'] == 2),
                               max(e['ts'] + e['dur'] for e in events if e['pid'] == 1))
            self.assertTrue(all(e['args']['status'] == 'interrupted' for e in events
                                if e['name'] == 'handle_contribution'))
            summary.write_text(json.dumps(dict(mode='ssh', nodes=nodes)))
            with self.assertRaisesRegex(ValueError, 'remote clocks'):
                build_trace(directory)
            single = build_trace(directory, node='worker')
            slices = [event for event in single['traceEvents'] if event['ph'] == 'X']
            self.assertEqual(len(slices), 3)
            self.assertEqual({event['pid'] for event in slices}, {2})
            self.assertEqual(min(event['ts'] for event in slices), 0)
            with self.assertRaisesRegex(ValueError, 'unknown node'):
                build_trace(directory, node='missing')

    def test_async_edge_exchange_survives_pending_child_work(self):
        import numpy as np
        from test_async_protocol import initial, packet
        from test_server_protocol import server
        from protocol import encode_async_params
        from transport import receive_message

        with tempfile.TemporaryDirectory() as tmp, server(edge=True, asynchronous=True) as (edge, peers, parent):
            recorder = MetricsRecorder(Path(tmp) / 'edge', trace_duration_s=10)
            edge.recorder = recorder
            try:
                initial(peers)
                edge._accept_contribution('w0', packet(0))
                receive_message(parent)
                edge._accept_contribution('w1', packet(1))
                first_start = edge.upstream_span
                edge._dispatch_parent_push(encode_async_params(1, 1, np.zeros(1, dtype=np.float32)))
                receive_message(peers['w0'])
                receive_message(parent)
                self.assertGreater(edge.upstream_span, first_start)
                edge._dispatch_parent_push(encode_async_params(2, 2, np.zeros(1, dtype=np.float32)))
                receive_message(peers['w1'])
                self.assertIsNone(edge.upstream_span)
            finally:
                recorder.close()
            with (recorder.directory / 'spans.csv').open() as file:
                rows = list(csv.DictReader(file))
            exchanges = [row for row in rows if row['track'] == 'upstream']
            self.assertEqual([row['request'] for row in exchanges], ['1', '2'])
            self.assertTrue(all(row['status'] == 'complete' for row in exchanges))
            self.assertTrue({'decode_gradients', 'aggregate_gradients', 'encode_gradients',
                             'send_gradients', 'decode_load_parameters', 'encode_parameters',
                             'send_parameters'} <= {row['operation'] for row in rows})

    def test_root_snapshot_encoding_and_each_send_in_both_modes(self):
        import zmq
        from test_server_protocol import server
        from transport import receive_message

        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), tempfile.TemporaryDirectory() as tmp, server(asynchronous=asynchronous) as (root, peers, _):
                for peer in peers.values():
                    receive_message(peer)  # assignment
                    receive_message(peer)  # initial snapshot
                root._eval_push = root._router.context.socket(zmq.PUSH)
                root._eval_push.bind('inproc://trace-eval')
                evaluation = root._router.context.socket(zmq.PULL)
                evaluation.connect('inproc://trace-eval')
                evaluation.setsockopt(zmq.RCVTIMEO, 2000)
                recorder = MetricsRecorder(Path(tmp) / 'root', trace_duration_s=10)
                root.recorder = recorder
                try:
                    root._broadcast_snapshot()
                    for peer in peers.values():
                        receive_message(peer)
                    root._push_eval()
                    receive_message(evaluation)
                finally:
                    recorder.close()
                with (recorder.directory / 'spans.csv').open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual([r['operation'] for r in rows], [
                    'encode_parameters', 'send_parameters', 'send_parameters',
                    'encode_evaluation_snapshot', 'send_evaluation_snapshot'])
                self.assertEqual([r['peer'] for r in rows if r['operation'] == 'send_parameters'], ['w0', 'w1'])
                self.assertTrue(all(int(a['end_ns']) <= int(b['start_ns']) for a, b in zip(rows, rows[1:])))
