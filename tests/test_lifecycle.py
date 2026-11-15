"""Real ZeroMQ lifecycle checks; no fake sockets, clocks, or training libraries."""
import queue
import threading
import time
import unittest

import zmq

from lifecycle import Lifecycle, RunStopped, RunFailed, failure_report, merge_reports


class LifecycleTests(unittest.TestCase):
    def test_failure_takes_precedence_over_interruption(self):
        interrupted = failure_report('worker: Ctrl+C', interrupted=True)
        failed = failure_report('eval: invalid tensor')
        for first, second in ((interrupted, failed), (failed, interrupted)):
            outcome = RunFailed(merge_reports(first, second))
            self.assertFalse(outcome.interrupted)
            self.assertEqual(str(outcome), 'eval: invalid tensor')

    def test_startup_and_training_deadlines(self):
        context = zmq.Context()
        control = context.socket(zmq.ROUTER)
        control.bind('inproc://control')
        incoming = context.socket(zmq.PULL)
        incoming.bind('inproc://training')
        try:
            run = Lifecycle(control, 'root', 'root', ['root', 'worker'], 'worker',
                            dict(startup_s=0.03, training_s=0.03, shutdown_s=0.1))
            with self.assertRaises(TimeoutError):
                run.ready()
            run.begin_round()
            with self.assertRaises(TimeoutError):
                run.wait([incoming])
            run.timeouts['training_s'] = None
            run.begin_round()
            self.assertIsNone(run.deadline)
        finally:
            context.destroy(linger=0)

    def exchange(self, failure=None, early_stop=False, missing_ack=False, interrupt_root=False):
        context = zmq.Context()
        results = queue.Queue()
        started = time.monotonic()
        timeouts = dict(startup_s=2, training_s=None, shutdown_s=0.3 if missing_ack else 2)

        def participant(node_id):
            control = context.socket(zmq.ROUTER if node_id == 'root' else zmq.DEALER)
            if node_id == 'root':
                control.setsockopt(zmq.ROUTER_MANDATORY, 1)
                control.bind('inproc://control')
            else:
                control.setsockopt(zmq.IDENTITY, node_id.encode())
                control.connect('inproc://control')
            run = Lifecycle(control, node_id, 'root', ['root', 'worker', 'eval'], 'eval', timeouts)
            try:
                try:
                    run.ready()
                    run.begin_round()
                    if node_id == 'root':
                        if interrupt_root:
                            raise RunStopped(failure_report('root: Ctrl+C', interrupted=True))
                        if failure or early_stop:
                            run.wait([])
                    elif node_id == 'worker' and missing_ack:
                        return  # Simulate disappearance after successful startup.
                    elif node_id == failure:
                        raise ValueError('deliberate participant failure')
                    elif node_id == 'eval' and early_stop:
                        run.request_stop()
                    else:
                        run.wait([])
                except RunStopped as stop:
                    run.finish(stop.error)
                except Exception as error:
                    run.finish(failure_report(str(error)))
                    raise
                else:
                    run.finish()
                results.put((node_id, 'ok'))
            except Exception as error:
                results.put((node_id, str(error)))
            finally:
                control.close(0)

        threads = [threading.Thread(target=participant, args=(name,), daemon=True)
                   for name in ('root', 'worker', 'eval')]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            records = dict(results.get_nowait() for _ in range(results.qsize()))
            self.assertEqual(set(records), {'root', 'eval'} if missing_ack else {'root', 'eval', 'worker'})
            if failure:
                self.assertTrue(all('deliberate participant failure' in value for value in records.values()), records)
            elif missing_ack:
                self.assertTrue(all(value != 'ok' for value in records.values()), records)
                self.assertIn('missing acknowledgements', records['root'])
            else:
                self.assertEqual(set(records.values()), {'ok'}, records)
                self.assertLess(time.monotonic() - started, 2, 'shutdown waited out its full budget')
        finally:
            # All sockets are owned and closed by their threads.
            context.term()

    def test_acknowledged_completion_and_successful_early_stop(self):
        self.exchange()
        self.exchange(early_stop=True)

    def test_worker_and_evaluator_failure_propagate(self):
        for participant in ('worker', 'eval'):
            with self.subTest(participant=participant):
                self.exchange(failure=participant)

    def test_missing_shutdown_ack_is_bounded_failure(self):
        self.exchange(missing_ack=True)
        self.exchange(missing_ack=True, interrupt_root=True)


def run_small_runtime(config_path, node_id, results, interrupt_worker=False, small_workload=False, asynchronous=False, flat=False, capped=True):
    """Real runtime/ResNet/autograd; only the CIFAR data source is synthetic."""
    import torch
    import training
    from dist_train import cli
    import contextlib
    import io
    torch.set_num_threads(1)
    dataset = torch.utils.data.TensorDataset(torch.zeros(8, 1, 28, 28) if small_workload else torch.zeros(2, 3, 32, 32),
                                            torch.zeros(8 if small_workload else 2, dtype=torch.long))
    def load_data(*, root, download, name):
        assert root == "./test-data" and download is True
        assert name == ("mnist" if small_workload else "cifar10")
        if interrupt_worker and node_id == 'worker':
            # Exercise the runtime's actual SIGINT/KeyboardInterrupt handler.
            import os
            import signal
            os.kill(os.getpid(), signal.SIGINT)
        return dataset, dataset

    training.load_datasets = load_data
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        from pathlib import Path
        status = cli(['--config', config_path, '--node-id', node_id,
                      '--metrics-dir', str(Path(config_path).parent / f'{node_id}.metrics')])
    text = output.getvalue()
    assert 'Coordinated shutdown completed.' in text, text
    assert 'INCOMPLETE' not in text and 'Traceback' not in text, text
    if interrupt_worker:
        assert status == 130 and 'Run interrupted:' in text, text
        results.put((node_id, 'interrupted: clean shutdown'))
    else:
        assert status is None, (status, text)
        if node_id == 'root':
            assert ('[TRAIN] update=1 applied_samples=' if asynchronous else '[TRAIN] round=1 samples=') in text, text
            if asynchronous:
                assert 'discarded_contributions=' in text, text
                assert ('applied_contributions=1 applied_samples=2' if capped
                        else 'applied_contributions=4 applied_samples=6') in text, text
                if not capped:
                    assert 'discarded_contributions=0 discarded_samples=0 partial_discarded_samples=0' in text, text
        results.put((node_id, 'ok'))
    if status:
        raise SystemExit(status)


class RuntimeLifecycleTests(unittest.TestCase):
    def run_hierarchy(self, interrupt_worker=False, small_workload=False, asynchronous=False, flat=False, capped=True, metrics_enabled=True):
        import json
        import multiprocessing
        import socket
        import tempfile
        from pathlib import Path

        reservations = [socket.socket() for _ in range(4)]
        try:
            for sock in reservations:
                sock.bind(('127.0.0.1', 0))
            endpoints = [f'tcp://127.0.0.1:{sock.getsockname()[1]}' for sock in reservations]
        finally:
            for sock in reservations:
                sock.close()
        root, control, edge, evaluator = endpoints
        raw = dict(monitoring=dict(enabled=False), metrics=dict(enabled=metrics_enabled),
                   data=dict(root="./test-data", download=True),
                   timeouts=dict(startup_s=30, shutdown_s=10),
                   training=dict(batch_size=2, n_epochs=1, eval_interval=200, log_interval=1000),
                   eval=dict(accuracy_threshold=2, time_limit_s=60),
                   nodes=[
                       dict(id='root', role='root_ps', host='127.0.0.1', endpoint=root, stop_endpoint=control),
                       dict(id='edge', role='edge_ps', host='127.0.0.1', endpoint=edge, parent='root'),
                       dict(id='worker', role='worker', host='127.0.0.1', parent='edge'),
                       dict(id='eval', role='eval', host='127.0.0.1', endpoint=evaluator)])
        if small_workload:
            raw['workload'] = 'lenet_mnist'
            raw['training'].update(max_rounds=1, cpu_threads=1, num_workers=0)
            raw['data'].update(train_samples=6, eval_samples=3)
        if asynchronous:
            raw['mode'] = 'async'
            raw['training']['max_rounds'] = 1 if capped else None
            raw['nodes'].append(dict(id='worker2', role='worker', host='127.0.0.1', parent='edge'))
            if flat:
                raw['nodes'] = [node for node in raw['nodes'] if node['id'] != 'edge']
                for node in raw['nodes']:
                    if node['role'] == 'worker':
                        node['parent'] = 'root'
        spawn = multiprocessing.get_context('spawn')
        results = spawn.Queue()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(json.dumps(raw))
            processes = [spawn.Process(target=run_small_runtime, args=(str(path), node['id'], results, interrupt_worker, small_workload, asynchronous, flat, capped))
                         for node in raw['nodes']]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 60
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                codes = [process.exitcode for process in processes]
                if interrupt_worker:
                    self.assertEqual(codes, [130] * 4)
                    records = dict(results.get(timeout=2) for _ in processes)
                    self.assertTrue(all(value == 'interrupted: clean shutdown' for value in records.values()), records)
                else:
                    self.assertEqual(codes, [0] * len(processes))
                    self.assertEqual(dict(results.get(timeout=2) for _ in processes),
                                     {node['id']: 'ok' for node in raw['nodes']})
                    final = json.loads((Path(directory) / 'root.metrics' / 'result.json').read_text())
                    self.assertGreater(final['elapsed'], 0)
                    if not metrics_enabled:
                        self.assertEqual((final['version'], final['samples']), (1, 2))
                        self.assertFalse(list(Path(directory).glob('*.metrics/*.csv')))
                        return
                    import csv
                    with (Path(directory) / 'root.metrics' / 'updates.csv').open() as file:
                        updates = list(csv.DictReader(file))
                    with (Path(directory) / 'eval.metrics' / 'evaluations.csv').open() as file:
                        evaluations = list(csv.DictReader(file))
                    self.assertTrue(updates and evaluations)
                    self.assertEqual(len(updates), int(updates[-1]['version']))
                    self.assertEqual(evaluations[-1]['samples'], updates[-1]['samples'])
                    self.assertEqual(evaluations[-1]['version'], updates[-1]['version'])
                    self.assertEqual(evaluations[-1]['root_elapsed'], updates[-1]['elapsed'])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)
                    process.close()
                results.close()
                results.join_thread()

    def test_hierarchical_runtime_completes_final_evaluation_and_shutdown(self):
        self.run_hierarchy()

    def test_worker_interrupt_fails_all_runtime_participants(self):
        self.run_hierarchy(interrupt_worker=True)

    def test_mnist_round_limit_and_final_evaluation(self):
        self.run_hierarchy(small_workload=True)

    def test_async_flat_and_hierarchy_completion_and_cap(self):
        for flat, capped in ((True, False), (False, False), (False, True)):
            with self.subTest(flat=flat, capped=capped):
                self.run_hierarchy(small_workload=True, asynchronous=True, flat=flat, capped=capped)

    def test_metrics_disabled_keeps_exact_final_result(self):
        self.run_hierarchy(small_workload=True, metrics_enabled=False)
