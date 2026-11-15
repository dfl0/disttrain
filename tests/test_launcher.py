"""Test process cleanup and logging without an SSH server."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from launcher import EXIT_MARKER, remote_result

LAUNCHER = str(Path(__file__).resolve().parents[1] / 'launcher.py')
# minimal child program for launcher tests
PROGRAM = '''
import argparse, json, signal, time
p = argparse.ArgumentParser()
p.add_argument('--config')
p.add_argument('--node-id')
a = p.parse_args()
cfg = json.load(open(a.config))
print('started ' + a.node_id, flush=True)
if cfg.get('fail') and a.node_id == 'worker':
    raise SystemExit(7)
if cfg.get('hold'):
    if cfg.get('ignore_signals'):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGINT, lambda *args: exit(130))
    while True:
        time.sleep(0.1)
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'dist_train.py').write_text(PROGRAM)
        self.config = self.root / 'input.yaml'
        self.raw = dict(timeouts=dict(shutdown_s=0.1), nodes=[
            dict(id='root', role='root_ps', host='localhost', endpoint='tcp://localhost:5550',
                 stop_endpoint='tcp://localhost:5551'),
            dict(id='eval', role='eval', host='localhost', endpoint='tcp://localhost:5552'),
            dict(id='worker', role='worker', host='localhost', parent='root')])

    def command(self, name):
        self.config.write_text(json.dumps(self.raw))
        return [sys.executable, LAUNCHER, str(self.config), name, '--mode', 'local',
                '--workdir', str(self.root), '--output-dir', str(self.root / 'logs')]

    def summary(self, name):
        return json.loads((self.root / 'logs' / name / 'summary.json').read_text())

    def test_local_completion_logs_config_and_no_overwrite(self):
        result = subprocess.run(self.command('done'), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = self.summary('done')
        self.assertEqual(summary['outcome'], 'success')
        self.assertTrue(all(n['exit_code'] == 0 for n in summary['nodes'].values()))
        self.assertEqual((self.root / 'logs/done/config.yaml').read_text(), self.config.read_text())
        for node in ('root', 'eval', 'worker'):
            self.assertIn('started ' + node, (self.root / f'logs/done/{node}.log').read_text())
        repeated = subprocess.run(self.command('done'), capture_output=True, text=True, timeout=10)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(self.summary('done'), summary)

    def test_failure_escalates_only_owned_processes(self):
        self.raw.update(hold=True, ignore_signals=True, fail=True)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     start_new_session=True)
        try:
            result = subprocess.run(self.command('failed'), capture_output=True, text=True, timeout=12)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            summary = self.summary('failed')
            self.assertEqual(summary['nodes']['worker']['exit_code'], 7)
            self.assertEqual(summary['nodes']['root']['exit_code'], -signal.SIGKILL)
            self.assertIsNone(unrelated.poll())
            self.assertEqual(summary['outcome'], 'failed')
        finally:
            unrelated.kill()
            unrelated.wait(timeout=5)

    def test_ctrl_c_stops_local_run(self):
        self.raw.update(hold=True)
        process = subprocess.Popen(self.command('interrupted'), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                log = self.root / 'logs/interrupted/worker.log'
                if log.exists() and 'started worker' in log.read_text():
                    break
                time.sleep(0.02)
            else:
                self.fail('nodes did not start')
            process.send_signal(signal.SIGINT)
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130, output)
            summary = self.summary('interrupted')
            self.assertEqual(summary['outcome'], 'interrupted')
            self.assertTrue(all(n['cleanup_confirmed'] for n in summary['nodes'].values()))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

    def test_remote_supervisor_eof_cleanup_and_confirmation(self):
        self.raw.update(hold=True)
        log = self.root / 'remote.log'
        request = dict(token='test-token', config=json.dumps(self.raw), node_id='worker',
                       python=sys.executable, shutdown_s=0.1)
        with log.open('wb') as output:
            process = subprocess.Popen([sys.executable, LAUNCHER, '--remote-node'], cwd=self.root,
                                       stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT)
            try:
                # simulate an SSH disconnect by closing stdin
                process.communicate(json.dumps(request).encode() + b'\n', timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
        report = remote_result(log, 'test-token')
        self.assertTrue(report['cleanup_confirmed'], log.read_text())
        self.assertIsNotNone(report['exit_code'])
        self.assertFalse(remote_result(log, 'wrong-token')['cleanup_confirmed'])
        self.assertIn(EXIT_MARKER, log.read_text())
