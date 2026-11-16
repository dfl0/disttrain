"""Test clock alignment and calibration messages."""
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest

import yaml
import zmq

from clock_alignment import ClockAlignment
from export_trace import build_trace
from analyze_runs import read_run, render_report
from lifecycle import Lifecycle, RunStopped
from metrics import MetricsRecorder


class ClockTests(unittest.TestCase):
    def test_mapping_report_and_trace_share_root_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            nodes = {'root': dict(role='root_ps'), 'worker': dict(role='worker'), 'eval': dict(role='eval')}
            (path / 'summary.json').write_text(json.dumps(dict(mode='ssh', outcome='success', nodes=nodes)))
            (path / 'config.yaml').write_text(yaml.safe_dump(dict(nodes=[dict(id=k, **v) for k, v in nodes.items()])))
            root = MetricsRecorder(path / 'root.metrics')
            root.record('clock_origins', 'training', 10_000_000_000, 0)
            root.record('updates', 1., 1, 32, 1, 32, 1, 0, 0)
            for node in ('worker', 'eval'):
                # local clock is 2 seconds ahead with a 100 ppm correction
                root.record('clock_calibration', node, 'startup', 12_000_000_000, 10_000_000_000, 1000)
                root.record('clock_calibration', node, 'shutdown', 22_000_000_000, 20_001_000_000, 2000)
            root.close()
            worker = MetricsRecorder(path / 'worker.metrics')
            worker.record('clock_origins', 'runtime', 12_000_000_000, 0)
            worker.record('resources', 5., .5, 5.01, 30., None, 20., 1000, None, None, 'all', 10., 20.)
            worker.record('spans', 'compute', 'main', 17_000_000_000, 17_100_000_000, 'complete', 1, 1, 'root')
            worker.close()
            evaluator = MetricsRecorder(path / 'eval.metrics')
            evaluator.record('clock_origins', 'evaluation', 12_000_000_000, 0)
            evaluator.record('evaluations', 1, 32, 1., 2., 5., 50., 1., 32)
            evaluator.close()
            clocks = ClockAlignment(path, nodes)
            timestamp, error, outside = clocks.project('worker', 17_000_000_000)
            self.assertAlmostEqual(timestamp, 5.0005)
            self.assertAlmostEqual(error, .0000015)
            self.assertFalse(outside)
            self.assertTrue(clocks.project('worker', 23_000_000_000)[2])
            self.assertAlmostEqual(clocks.describe('worker')['correction_ppm'], 100.)
            trace = build_trace(path)
            span = next(e for e in trace['traceEvents'] if e['ph'] == 'X')
            self.assertAlmostEqual(span['ts'] / 1e6, timestamp)
            self.assertEqual(span['args']['local_duration_us'], 100000.)
            run = read_run(path)
            self.assertAlmostEqual(run['resources'][0]['aligned_s'], timestamp)
            self.assertAlmostEqual(run['evaluations'][0]['done_root_s'], timestamp)
            self.assertAlmostEqual(run['evaluations'][0]['result_delay_s'], 4.0005)
            html = render_report([run])
            self.assertIn('Clock alignment', html)
            self.assertIn('Throughput and device CPU', html)
            self.assertIn('Evaluation result delay', html)
            # one sample provides an offset but no drift estimate
            clocks.anchors['worker'].pop('shutdown')
            self.assertIn('offset only', clocks.describe('worker')['method'])
            clocks.anchors.pop('worker')
            self.assertIsNone(clocks.project('worker', 17_000_000_000))

    def test_calibration_uses_existing_control_and_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            recorder = MetricsRecorder(path / 'root.metrics')
            recorder.clock_origin('training', time.perf_counter())
            context = zmq.Context()
            errors = queue.Queue()
            def participant(node):
                sock = context.socket(zmq.ROUTER if node == 'root' else zmq.DEALER)
                if node == 'root':
                    sock.bind('inproc://clock')
                else:
                    sock.setsockopt(zmq.IDENTITY, node.encode())
                    sock.connect('inproc://clock')
                run = Lifecycle(sock, node, 'root', ['root', 'worker'], 'worker',
                                dict(startup_s=3, shutdown_s=3, training_s=None))
                if node == 'root':
                    run.clock_recorder = recorder
                try:
                    run.ready()
                    run.begin_round()
                    if node != 'root':
                        try:
                            run.wait([])
                        except RunStopped:
                            pass
                    run.finish()
                except BaseException as exc:
                    errors.put(exc)
                finally:
                    sock.close(0)
            threads = [threading.Thread(target=participant, args=(n,), daemon=True) for n in ('root', 'worker')]
            try:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(5)
                self.assertFalse(any(t.is_alive() for t in threads))
                if not errors.empty():
                    raise errors.get()
                recorder.close()
                clocks = ClockAlignment(path, {'root': dict(role='root_ps'), 'worker': dict(role='worker')})
                self.assertEqual(set(clocks.anchors['worker']), {'startup', 'shutdown'})
                for anchor in clocks.anchors['worker'].values():
                    self.assertLessEqual(abs(anchor['local_ns'] - anchor['root_ns']), anchor['uncertainty_ns'])
            finally:
                context.destroy(linger=0)
                if not recorder.closed:
                    recorder.close()
