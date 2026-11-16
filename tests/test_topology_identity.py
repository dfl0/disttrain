"""Test topology validation and runtime setup."""
import subprocess
import json
import tempfile
from pathlib import Path
import sys
import unittest
from topology import build_topology, load_topology
from runtime import determine_role, main
from protocol import Role
from launcher import launch_entries


def flat():
    return dict(nodes=[
        dict(id='root', role='root_ps', host='127.0.0.1', endpoint='tcp://127.0.0.1:5550', stop_endpoint='tcp://127.0.0.1:5551'),
        dict(id='eval', role='eval', host='127.0.0.1', endpoint='tcp://127.0.0.1:5560'),
        dict(id='worker-1', role='worker', host='127.0.0.1', parent='root'),
        dict(id='worker-2', role='worker', host='127.0.0.1', parent='root'),
    ])


class TopologyIdentityTests(unittest.TestCase):
    def test_metric_recording_controls(self):
        self.assertEqual(build_topology(flat())['metrics'],
                         dict(enabled=True, update_every=1, measure_overhead=True, trace_duration_s=0))
        configured = dict(enabled=True, update_every=7, measure_overhead=False, trace_duration_s=5)
        self.assertEqual(build_topology({**flat(), 'metrics': configured})['metrics'], configured)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                build_topology({**flat(), 'metrics': dict(update_every=invalid)})

        for invalid in (-1, True, float('nan'), float('inf'), '5'):
            with self.assertRaises(ValueError):
                build_topology({**flat(), 'metrics': dict(trace_duration_s=invalid)})
        with self.assertRaises(ValueError):
            build_topology({**flat(), 'metrics': dict(enabled=False, trace_duration_s=5)})

    def test_async_mode_and_trace_configuration(self):
        raw = flat()
        self.assertEqual(build_topology(raw)['mode'], 'sync')
        raw.update(mode='async', logging=dict(trace_communication=True))
        topology = build_topology(raw)
        self.assertFalse(topology['node_configs']['root']['sync'])
        self.assertTrue(topology['trace_communication'])
        raw['nodes'][0]['sync'] = True
        with self.assertRaises(ValueError):
            build_topology(raw)
        raw['nodes'][0].pop('sync')
        raw["nodes"][0]["mode"] = "sync"
        self.assertTrue(build_topology(raw)["fully_synchronous"])
        raw['logging']['trace_communication'] = 'yes'
        with self.assertRaises(ValueError):
            build_topology(raw)

    def test_same_host_flat_and_hierarchical_ids_do_not_overwrite(self):
        raw = flat()
        topology = build_topology(raw)
        self.assertEqual(topology['node_configs']['root']['children'], ['worker-1', 'worker-2'])
        self.assertEqual(determine_role('worker-2', topology)['role'], Role.WORKER)
        self.assertEqual(topology['node_configs']['worker-1']['parent_endpoint'], 'tcp://127.0.0.1:5550')
        ranks = topology['worker_ranks']
        self.assertEqual(ranks, {'worker-1': 0, 'worker-2': 1})
        for index, worker in enumerate(raw['nodes'][2:]):
            edge_id = f'edge-{index}'
            worker['parent'] = edge_id
            raw['nodes'].append(dict(id=edge_id, role='edge_ps', host='127.0.0.1', parent='root',
                                    endpoint=f'tcp://127.0.0.1:{5570 + index}', bind_endpoint=f'tcp://*:{5570 + index}'))
        topology = build_topology(raw)
        self.assertEqual(topology['worker_ranks'], ranks)
        self.assertEqual(build_topology({**raw, 'nodes': list(reversed(raw['nodes']))})['worker_ranks'], ranks)
        self.assertEqual(len(topology['node_configs']), 6)
        self.assertEqual(topology['node_configs']['root']['children'], ['edge-0', 'edge-1'])
        self.assertEqual(topology['node_configs']['edge-1']['children'], ['worker-2'])
        self.assertEqual(determine_role('edge-0', topology)['role'], Role.EDGE_PS)
        self.assertEqual(topology['worker_specs']['worker-2']['parent_endpoint'], 'tcp://127.0.0.1:5571')

    def test_invalid_ids_hosts_endpoints_and_membership(self):
        mutations = [
            lambda r: r['nodes'][2].update(cpu=50),
            lambda r: r['nodes'][2].update(bw='10mbit'),
            lambda r: r['nodes'][0].update(sync=False),
            lambda r: r["nodes"][0].update(mode="invalid"),
            lambda r: r["nodes"][2].update(mode="async"),
            lambda r: r.update(training={'local_steps': 0}),
            lambda r: r['nodes'][2].update(worker_rank=9),
            lambda r: r['nodes'].append(dict(r['nodes'][2])),
            lambda r: r['nodes'][2].update(id='bad id'),
            lambda r: r['nodes'][2].update(host='-host'),
            lambda r: r['nodes'][2].update(host='host;cmd'),
            lambda r: r['nodes'][0].update(endpoint='tcp://*:5550'),
            lambda r: r['nodes'][0].update(endpoint='tcp://0.0.0.0:5550'),
            lambda r: r['nodes'][0].update(endpoint='tcp://host:70000'),
            lambda r: r['nodes'][0].update(endpoint='tcp://user@host:5550'),
            lambda r: r['nodes'][0].update(endpoint='tcp://host:5550/path'),
            lambda r: r['nodes'][0].update(endpoint='tcp://host:5550?q=1'),
            lambda r: r['nodes'][1].update(endpoint='tcp://127.0.0.1:5550'),
            lambda r: r['nodes'][1].update(endpoint='tcp://different-host:5560', bind_endpoint='tcp://*:5550'),
            lambda r: r['nodes'][2].update(parent='missing'),
            lambda r: r['nodes'][2].update(parent='eval'),
            lambda r: r['nodes'][0].pop('stop_endpoint'),
            lambda r: r['nodes'].pop(1),
        ]
        for mutate in mutations:
            raw = flat(); mutate(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError): build_topology(raw)
        with self.assertRaisesRegex(ValueError, 'nodes list'):
            build_topology({'bind_addr': 'localhost'})

    def test_unknown_id_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text(json.dumps(flat()))
            with self.assertRaisesRegex(ValueError, 'Unknown node ID'):
                main(str(config), node_id='not-a-node')

    def test_cli_help_and_required_arguments(self):
        result = subprocess.run([sys.executable, 'dist_train.py', '--help'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--config', result.stdout)
        self.assertIn('--node-id', result.stdout)
        for args, missing in ((['--node-id', 'root'], '--config'),
                              (['--config', 'unused.yaml'], '--node-id')):
            with self.subTest(missing=missing):
                result = subprocess.run([sys.executable, 'dist_train.py', *args], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn(missing, result.stderr)

    def test_example_config(self):
        topology = load_topology('configs/example_config.yaml')
        self.assertIn(topology['root_id'], topology['node_configs'])

    def test_colocated_nodes_launch_individually_in_role_order(self):
        topology = build_topology({
            'nodes': [
                {'id': 'worker-1', 'role': 'worker', 'host': 'host-a', 'parent': 'edge'},
                {'id': 'eval', 'role': 'eval', 'host': 'host-a', 'endpoint': 'tcp://host-a:5560'},
                {'id': 'edge', 'role': 'edge_ps', 'host': 'host-a', 'parent': 'root',
                 'endpoint': 'tcp://host-a:5552'},
                {'id': 'root', 'role': 'root_ps', 'host': 'host-a',
                 'endpoint': 'tcp://host-a:5550', 'stop_endpoint': 'tcp://host-a:5551'},
                {'id': 'worker-2', 'role': 'worker', 'host': 'host-a', 'parent': 'edge'},
            ],
        })
        self.assertEqual(launch_entries(topology), [
            ('host-a', 'root', 'root_ps'),
            ('host-a', 'edge', 'edge_ps'),
            ('host-a', 'eval', 'eval'),
            ('host-a', 'worker-1', 'worker'),
            ('host-a', 'worker-2', 'worker'),
        ])


    def test_runtime_bind_failure_exits_without_resource_commands_or_monitor_leak(self):
        import zmq
        context = zmq.Context()
        blocker = context.socket(zmq.ROUTER)
        port = blocker.bind_to_random_port('tcp://127.0.0.1')
        try:
            raw = flat()
            raw['timeouts'] = dict(shutdown_s=0.1)
            raw['nodes'][0]['endpoint'] = f'tcp://127.0.0.1:{port}'
            with tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / 'config.yaml'
                config.write_text(json.dumps(raw))  # JSON is valid YAML.
                script = """
import sys, threading, zmq
from runtime import main
try:
    main(sys.argv[1], node_id='root')
except zmq.ZMQError as error:
    if error.errno != zmq.EADDRINUSE:
        raise
else:
    raise AssertionError('expected occupied endpoint to reject startup')
assert not any(t.name.startswith('disttrain-metrics-') for t in threading.enumerate())
"""
                result = subprocess.run([sys.executable, '-c', script, str(config)],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            context.destroy(linger=0)

    def test_monitoring_configuration_and_real_samples(self):
        import threading
        import time
        from runtime import RuntimeContext, collect_sys_metrics

        raw = flat()
        raw['monitoring'] = dict(enabled=False, interval_s=0.04, interface=None, console=False)
        self.assertEqual(build_topology(raw)['monitoring'], raw['monitoring'])
        for invalid in (None, {'enabled': 1}, {'interval_s': 0},
                        {'interval_s': float('nan')}, {'interface': ''}, {'typo': 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_topology({**raw, 'monitoring': invalid})

        stop = threading.Event()
        samples = []
        context = RuntimeContext('test-host', Role.WORKER, time.perf_counter())

        def record(message):
            samples.append(json.loads(message.removeprefix('[METRICS] ')))
            if len(samples) == 2:
                stop.set()

        thread = threading.Thread(target=collect_sys_metrics,
                                  args=(stop, context, record),
                                  kwargs=dict(interval=0.01, interface='disttrain-nonexistent', console=True))
        try:
            thread.start()
            thread.join(timeout=5)
        finally:
            stop.set()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertGreater(sample['interval_s'], 0)
            self.assertGreater(sample['process_rss_bytes'], 0)
            self.assertIsNone(sample['host_network_tx_bytes_per_s'])
            self.assertGreaterEqual(sample['sample_end_s'], sample['t_s'])
        self.assertGreater(samples[1]['t_s'], samples[0]['t_s'])
        # A stop requested before startup must not wait for the sampling interval.
        collect_sys_metrics(stop, context, record, interval=60)
        self.assertEqual(len(samples), 2)

    def test_lifecycle_timeout_configuration(self):
        self.assertEqual(build_topology(flat())['timeouts'],
                         dict(startup_s=120, training_s=None, shutdown_s=10))
        configured = dict(startup_s=5, training_s=60, shutdown_s=2)
        self.assertEqual(build_topology({**flat(), 'timeouts': configured})['timeouts'], configured)
        for invalid in (None, {'unknown': 1}, {'startup_s': None}, {'startup_s': 0},
                        {'training_s': -1}, {'shutdown_s': True}, {'training_s': float('inf')}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_topology({**flat(), 'timeouts': invalid})

    def test_dataset_configuration(self):
        self.assertEqual(build_topology(flat())['data'], dict(root='/mnt/data', download=False, train_samples=None, eval_samples=None))
        custom = dict(root='~/datasets/cifar10', download=True)
        self.assertEqual(build_topology({**flat(), 'data': custom})['data'], {**custom, 'train_samples': None, 'eval_samples': None})
        for invalid in (None, {'root': ''}, {'root': 123}, {'download': 'yes'}, {'typo': True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_topology({**flat(), 'data': invalid})

    def test_small_workload_settings(self):
        raw = flat()
        raw.update(workload='lenet_mnist', training=dict(max_rounds=3, num_workers=0, cpu_threads=1),
                   data=dict(train_samples=100, eval_samples=20))
        self.assertEqual(build_topology(raw)['training']['max_rounds'], 3)
        for field, values in [('workload', 'unknown'), ('training', {'max_rounds': 0}),
                              ('training', {'num_workers': -1}), ('training', {'log_interval': 0}), ('training', {'cpu_threads': True}),
                              ('data', {'eval_samples': 0}), ('data', {'train_samples': 1.5})]:
            with self.subTest(field=field, values=values), self.assertRaises(ValueError):
                build_topology({**raw, field: values})
