"""Test CSV analysis and report generation."""
from pathlib import Path
import tempfile
import unittest
import yaml
from analyze_runs import read_run, interval_points, render_report
from metrics import MetricsRecorder


class AnalysisTests(unittest.TestCase):
    def test_snapshot_metadata_and_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.yaml').write_text(yaml.safe_dump(dict(mode='async', nodes=[
                dict(id='root', role='root_ps'), dict(id='eval', role='eval')])))
            root = MetricsRecorder(path / 'root.metrics')
            root.record('updates', 1.0000123, 10, 320, 10, 32, 1, 0, 0)
            root.record('updates', 2.0000123, 20, 672, 21, 64, 2, 320, 2)
            root.record('accounting', 21, 672, 1, 32, 4)
            root.record('resources', 1.5, .5, 1.51, 30, None, 20, 1048576, None, None, 'all', None, None)
            root.close()
            evaluator = MetricsRecorder(path / 'eval.metrics')
            evaluator.record('evaluations', 15, 480, 1.23456789, 1.5, 1.7, 72, .8, 100)
            evaluator.close()
            run = read_run(path)
            self.assertEqual(run['evaluations'][0]['samples'], 480)  # no root version-15 record needed
            self.assertEqual(run['evaluations'][0]['root_elapsed'], 1.23456789)
            self.assertEqual(run['accounting']['partial_discarded_samples'], 4)
            self.assertEqual(len(run['resources']), 1)
            self.assertTrue(any('differs' in w for w in run['warnings']))
            intervals = interval_points(run['progress'])
            self.assertAlmostEqual(intervals[1]['throughput'], 352)
            self.assertEqual(intervals[1]['aggregate'], 1.1)
            run['warnings'].append('<script>untrusted()</script>')
            report = render_report([run], display_interval_s=0.5)
            self.assertIn('&lt;script&gt;untrusted()&lt;/script&gt;', report)
            self.assertIn('Process resident memory', report)
            self.assertIn('plotly.js', report)
            self.assertIn('0.5 s display intervals', report)
            self.assertIn('Measurement overhead', report)
            self.assertIn('record:updates', report)
            self.assertNotIn('<script src=', report)

    def test_zero_duration_and_counter_regression_are_not_rates(self):
        points = interval_points([dict(version=1, samples=32, elapsed=0, contributions=1),
                                  dict(version=2, samples=64, elapsed=0, contributions=2),
                                  dict(version=1, samples=32, elapsed=1, contributions=1)])
        self.assertTrue(all(p['throughput'] is None for p in points))

    def test_display_grouping_and_invalid_intervals(self):
        progress = [dict(version=i, samples=i * 32, contributions=i, elapsed=i * .125) for i in range(1, 6)]
        grouped = interval_points(progress, .5)
        self.assertEqual([p['elapsed'] for p in grouped], [.5, .625])
        self.assertTrue(all(p['throughput'] == 256 for p in grouped))
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                interval_points(progress, invalid)

    def test_explicit_baseline_repeats_and_work_matching(self):
        import copy
        import json
        from analyze_runs import compare_instrumentation
        with tempfile.TemporaryDirectory() as directory:
            runs = []
            for name, enabled, training_s, wall_s in [('off1', False, 10, 20), ('off2', False, 12, 22),
                                                     ('on1', True, 15, 30), ('on2', True, 18, 33)]:
                path = Path(directory) / name
                path.mkdir()
                (path / 'config.yaml').write_text(yaml.safe_dump(dict(
                    nodes=[dict(id='root', role='root_ps')], metrics=dict(enabled=enabled),
                    monitoring=dict(enabled=False))))
                (path / 'summary.json').write_text(json.dumps(dict(
                    outcome='success', mode='local', started_at=100, finished_at=100 + wall_s)))
                folder = path / 'root.metrics'
                if enabled:
                    recorder = MetricsRecorder(folder)
                    recorder.record('updates', training_s, 3, 96, 3, 32, 1, 0, 0)
                    recorder.close()
                else:
                    folder.mkdir()
                (folder / 'result.json').write_text(json.dumps(dict(version=3, samples=96, contributions=3, elapsed=training_s)))
                runs.append(read_run(path))
            baselines = [run['path'] for run in runs[:2]]
            groups = compare_instrumentation(runs, baselines)
            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[0]['training']['median'], 11)
            self.assertEqual(groups[1]['count'], 2)
            self.assertEqual(groups[1]['training']['slowdown'], 50)
            self.assertEqual(groups[1]['launcher']['slowdown'], 50)
            self.assertGreater(runs[2]['recording_windows']['root'], 0)
            report = render_report(runs, baseline_paths=baselines)
            self.assertIn('Recording-time fraction (not slowdown)', report)
            self.assertIn('Observed training slowdown', report)
            self.assertIn('not confidence intervals', report)
            bad = copy.deepcopy(runs)
            bad[-1]['final_result']['samples'] = 95
            with self.assertRaisesRegex(ValueError, 'equal applied-sample'):
                compare_instrumentation(bad, baselines)
            bad[-1]['final_result']['samples'] = 96
            bad[-1]['config']['training'] = dict(batch_size=999)
            with self.assertRaisesRegex(ValueError, 'matching config'):
                compare_instrumentation(bad, baselines)
