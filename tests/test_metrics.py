"""Test CSV recording and resource sampling."""
import csv
from pathlib import Path
import tempfile
import threading
import time
import unittest
from metrics import MetricsRecorder
from runtime import RuntimeContext, collect_sys_metrics
from protocol import Role


class MetricsTests(unittest.TestCase):
    def test_bounded_writer_drains_all_rows_from_concurrent_producers(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MetricsRecorder(Path(directory) / 'metrics', queue_size=1)
            def produce(offset):
                for i in range(300):
                    recorder.record('updates', i / 1000000, offset + i, i, i, 1, 1, 0, 0)
            threads = [threading.Thread(target=produce, args=(offset,)) for offset in (0, 1000)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
                self.assertFalse(thread.is_alive())
            recorder.close()
            with (recorder.directory / 'updates.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 600)
            self.assertEqual(len({row['version'] for row in rows}), 600)
            with (recorder.directory / 'recording.csv').open() as file:
                stats = next(csv.DictReader(file))
            self.assertEqual(stats['records'], '600')
            self.assertEqual(stats['complete'], 'True')
            with (recorder.directory / 'overhead.csv').open() as file:
                overhead = {row['component']: row for row in csv.DictReader(file)}
            self.assertEqual(overhead['record:updates']['calls'], '600')
            self.assertGreater(float(overhead['record:updates']['wall_s']), 0)
            self.assertGreater(float(overhead['writer']['cpu_s']), 0)

    def test_io_failure_surfaces_without_queue_deadlock(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MetricsRecorder(Path(directory) / 'metrics', queue_size=1)
            (recorder.directory / 'updates.csv').mkdir()  # a directory cannot be opened as an output file
            recorder.record('updates', 0, 1, 1, 1, 1, 1, 0, 0)
            with self.assertRaisesRegex(RuntimeError, 'recording failed'):
                recorder.close()

    def test_resource_recording_without_console_output(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MetricsRecorder(Path(directory) / 'metrics')
            stop = threading.Event()
            output = []
            context = RuntimeContext('local', Role.WORKER, time.perf_counter())
            thread = threading.Thread(target=collect_sys_metrics, args=(stop, context, output.append, .01),
                                      kwargs=dict(recorder=recorder, console=False))
            thread.start()
            time.sleep(.15)
            stop.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            recorder.close()
            with (recorder.directory / 'resources.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(rows)
            self.assertEqual(output, [])
            self.assertGreater(float(rows[0]['interval_s']), 0)

    def test_update_sampling_keeps_final_without_duplicates(self):
        for count, expected in ((7, [3, 6, 7]), (6, [3, 6]), (2, [2])):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                recorder = MetricsRecorder(Path(directory) / 'metrics', update_every=3, measure_overhead=False)
                for version in range(1, count + 1):
                    recorder.update(version / 1000, version, version * 32, version, 32, 1, 0, 0)
                recorder.close()
                recorder.close()  # closing twice must not duplicate the final record
                with (recorder.directory / 'updates.csv').open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual([int(row['version']) for row in rows], expected)
                self.assertEqual(int(rows[-1]['samples']), count * 32)
                self.assertFalse((recorder.directory / 'overhead.csv').exists())
