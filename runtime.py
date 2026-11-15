"""Per-run setup/cleanup and telemetry; resource throttling is temporarily disabled."""
import time
import json
import shutil
import socket
import subprocess
import threading
from dataclasses import dataclass

from protocol import Role, ParamSerializer
from topology import load_topology


@dataclass
class RuntimeContext:
    hostname: str
    role: Role
    start_time: float
    node_id: str = None

    def log(self, msg, indent=0):
        elapsed = time.perf_counter() - self.start_time
        whole = int(elapsed)
        ms = int((elapsed - whole) * 1000)
        timestamp = time.strftime("%H:%M:%S", time.gmtime(whole))
        timestamp = f"{timestamp}:{ms:03d}"
        assert self.role
        short_host = self.node_id or (self.hostname.split("-", 1)[-1] if "-" in self.hostname else self.hostname)
        prefix = f"({timestamp}) {short_host:<12} [{self.role.name:<8}]  "
        indent_str = "    " * indent
        print(f"{prefix}{indent_str}{msg}", flush=True)




def determine_role(node_id, topology):
    cfg = topology["node_configs"].get(node_id)
    if cfg is None:
        raise ValueError(f"Unknown node ID {node_id!r}; configured IDs: {list(topology['node_configs'])}")
    result = dict(cfg)
    result["role"] = Role[cfg["role"].upper()]
    return result


def collect_sys_metrics(stop_event, context, log, interval, interface=None, recorder=None, console=False):
    """Periodic host observations, not operation timings. None means unavailable."""
    import psutil

    def observe(read):
        try:
            return read()
        except (AttributeError, NotImplementedError, OSError, psutil.Error):
            return None

    def net_bytes():
        if interface is None:
            return psutil.net_io_counters()
        return psutil.net_io_counters(pernic=True).get(interface)

    process = psutil.Process()
    throttle_command = shutil.which('vcgencmd')
    previous_net = observe(net_bytes)
    previous_time = time.perf_counter()
    observe(lambda: psutil.cpu_percent(interval=None))  # prime the CPU measurement
    while not stop_event.wait(interval):
        sample_started = time.perf_counter() if recorder and recorder.measure_overhead else None
        current_net = observe(net_bytes)
        now = time.perf_counter()
        elapsed = now - previous_time
        cpu = observe(lambda: psutil.cpu_percent(interval=None))
        tx = rx = None
        if previous_net is not None and current_net is not None:
            sent = current_net.bytes_sent - previous_net.bytes_sent
            received = current_net.bytes_recv - previous_net.bytes_recv
            if sent >= 0 and received >= 0:
                tx, rx = sent / elapsed, received / elapsed
        previous_net, previous_time = current_net, now

        frequency = observe(lambda: psutil.cpu_freq())
        memory = observe(psutil.virtual_memory)
        rss = observe(lambda: process.memory_info().rss)
        temperatures = observe(lambda: psutil.sensors_temperatures())
        temperatures = ({group: [{'label': sensor.label, 'celsius': sensor.current}
                                 for sensor in sensors]
                         for group, sensors in temperatures.items()} if temperatures else None)
        throttled_now = throttled_before = None
        if throttle_command and not stop_event.is_set():
            try:
                result = subprocess.run([throttle_command, 'get_throttled'],
                                        capture_output=True, text=True, check=True, timeout=0.5)
                value = int(result.stdout.strip().split('=')[-1], 16)
                throttled_now = bool(value & (1 << 2))
                throttled_before = bool(value & (1 << 18))
            except (OSError, subprocess.SubprocessError, ValueError):
                # stop retrying a failing command
                throttle_command = None
        if stop_event.is_set():
            break
        values = (
            now - context.start_time, elapsed, time.perf_counter() - context.start_time,
            cpu, frequency.current if frequency else None,
            memory.percent if memory else None, rss, throttled_now, throttled_before,
            interface if interface is not None else 'all', tx, rx)
        if sample_started is not None:
            recorder.measure('resource_sampling', time.perf_counter() - sample_started)
        if recorder:
            recorder.record('resources', *values)
            for group, sensors in (temperatures or {}).items():
                for sensor in sensors:
                    recorder.record('temperatures', values[0], group, sensor['label'], sensor['celsius'])
        if console:
            from metrics import SCHEMAS
            log('[METRICS] ' + json.dumps(dict(zip(SCHEMAS['resources'], values))))





def main(config_file, *, node_id, metrics_dir=None):
    topology = load_topology(config_file)
    role_cfg = determine_role(node_id, topology)
    role = role_cfg["role"]
    training = topology["training"]
    from transport import open_role_sockets, open_control_socket
    from lifecycle import Lifecycle, RunStopped, RunFailed, failure_report
    from ps import ParameterServer
    from models import (
        LeNet, ResNet20CIFAR, BasicBlock,
        disable_state, init_params,
    )
    from training import run_worker, run_evaluator

    import zmq
    hostname = socket.gethostname()
    context = RuntimeContext(hostname, role, time.perf_counter(), node_id=node_id)
    log = context.log

    log(f"({hostname}) starting as {role.name}")

    ctx = zmq.Context()

    monitor_stop = threading.Event()
    monitor_thread = None
    lifecycle = None
    report = ''
    original_error = None
    server = None
    recorder = None
    monitor_errors = []
    try:
        control = open_control_socket(ctx, role, role_cfg, topology, node_id)
        lifecycle = Lifecycle(control, node_id, topology["root_id"],
                              topology["node_configs"], topology["eval_id"], topology["timeouts"])
        if topology['metrics']['enabled']:
            from metrics import MetricsRecorder
            from pathlib import Path
            import uuid
            destination = metrics_dir or str(Path('metrics') / f'{node_id}-{uuid.uuid4().hex[:12]}')
            recorder = MetricsRecorder(destination, update_every=topology['metrics']['update_every'],
                                       measure_overhead=topology['metrics']['measure_overhead'],
                                       trace_duration_s=topology['metrics']['trace_duration_s'])
            log(f'CSV metrics: {destination}')
            lifecycle.health_check = recorder.check
            lifecycle.clock_recorder = recorder
            recorder.clock_origin("runtime", context.start_time)
        sockets = open_role_sockets(ctx, role, role_cfg, topology, node_id)
        import torch
        torch.manual_seed(training["seed"])

        if training['cpu_threads'] is not None:
            torch.set_num_threads(training['cpu_threads'])
            torch.set_num_interop_threads(training['cpu_threads'])
        dataset_name = 'mnist' if topology['workload'] == 'lenet_mnist' else 'cifar10'
        model = (LeNet() if dataset_name == 'mnist'
                 else ResNet20CIFAR(BasicBlock, layers=[3, 3, 3], num_classes=10))
        model.apply(disable_state)

        serializer = ParamSerializer(model)

        monitoring = topology['monitoring']
        if monitoring['enabled'] and (recorder or monitoring['console']):
            def monitor():
                try:
                    collect_sys_metrics(monitor_stop, context, log, monitoring['interval_s'],
                                        monitoring['interface'], recorder, monitoring['console'])
                except Exception as error:
                    monitor_errors.append(error)
            monitor_thread = threading.Thread(target=monitor,
                name=f'disttrain-metrics-{node_id}', daemon=True)
            monitor_thread.start()
        server_type = ParameterServer
        server_options = {}
        if topology['mode'] == 'async':
            from async_ps import AsyncParameterServer
            server_type = AsyncParameterServer
            groups = {child: ({topology['worker_ranks'][child]} if child in topology['worker_ranks']
                              else {topology['worker_ranks'][worker] for worker in topology['node_configs'][child]['children']})
                      for child in role_cfg.get('children', [])}
            server_options = dict(worker_groups=groups, reference_samples=(len(topology['worker_specs'])
                                  * training['batch_size'] * training['local_steps']),
                                  trace=topology['trace_communication'])
        match role:
            case Role.ROOT_PS:
                server = server_type(
                    **server_options, recorder=recorder, worker_count=len(topology['worker_specs']),
                    model=model,
                    children_ids=role_cfg["children"],
                    child_ranks={child: topology["worker_ranks"].get(child, -1)
                                 for child in role_cfg["children"]},
                    sync=role_cfg["sync"],
                    router_sock=sockets.router,
                    stop_sock=sockets.stop,
                    eval_push_sock=sockets.evaluator,
                    serializer=serializer,
                    init_params=init_params,
                    parent=None,
                    lr=training["learning_rate"],
                    eval_interval=training["eval_interval"],
                    max_rounds=training["max_rounds"],
                    log_interval=training["log_interval"],
                    log=log,
                    node_id=node_id,
                    lifecycle=lifecycle,
                )
                server.run()

            case Role.EDGE_PS:
                server = server_type(
                    **server_options, recorder=recorder, worker_count=len(topology['worker_specs']),
                    model=model,
                    children_ids=role_cfg["children"],
                    child_ranks={child: topology["worker_ranks"].get(child, -1)
                                 for child in role_cfg["children"]},
                    sync=role_cfg["sync"],
                    router_sock=sockets.router,
                    dealer_sock=sockets.dealer,
                    serializer=serializer,
                    init_params=init_params,
                    parent=role_cfg["parent"],
                    lr=training["learning_rate"],
                    eval_interval=training["eval_interval"],
                    max_rounds=training["max_rounds"],
                    log_interval=training["log_interval"],
                    log=log,
                    node_id=node_id,
                    lifecycle=lifecycle,
                )
                server.run()

            case Role.WORKER:
                n_workers = len(topology["worker_specs"])
                run_worker(
                    recorder=recorder,
                    mode=topology['mode'], trace=topology['trace_communication'],
                    dataset_name=dataset_name, sample_limit=topology["data"]["train_samples"],
                    num_workers=training["num_workers"],
                    data_root=topology["data"]["root"], download=topology["data"]["download"],
                    dealer_sock=sockets.dealer,
                    model=model,
                    n_workers=n_workers,
                    batch_size=training["batch_size"],
                    n_epochs=training["n_epochs"],
                    local_steps=training["local_steps"],
                    worker_rank=role_cfg["worker_rank"],
                    seed=training["seed"],
                    log=log,
                    node_id=node_id,
                    lifecycle=lifecycle,
                )

            case Role.EVAL:
                run_evaluator(
                    recorder=recorder,
                    dataset_name=dataset_name, sample_limit=topology["data"]["eval_samples"],
                    num_workers=training["num_workers"], seed=training["seed"],
                    data_root=topology["data"]["root"], download=topology["data"]["download"],
                    pull_sock=sockets.pull,
                    push_sock=sockets.push,
                    model=model,
                    log=log,
                    accuracy_threshold=topology["eval_accuracy_threshold"],
                    time_limit_s=topology["eval_time_limit_s"],
                    lifecycle=lifecycle,
                    batch_size=training["batch_size"],
                )

    except RunStopped as stop:
        report = stop.error
        original_error = None
    except (Exception, KeyboardInterrupt) as error:
        original_error = error
        report = failure_report(
            f'{node_id}: Ctrl+C' if isinstance(error, KeyboardInterrupt)
            else f'{node_id}: {type(error).__name__}: {error}',
            interrupted=isinstance(error, KeyboardInterrupt))
    else:
        report = ''
        original_error = None
    finally:
        # stop sampling before closing the recorder
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2)
            if monitor_thread.is_alive():
                monitor_errors.append(TimeoutError('resource sampler did not stop'))
        try:
            if recorder:
                recorder.check()
            if monitor_errors:
                raise monitor_errors[0]
        except Exception as error:
            from lifecycle import merge_reports
            report = merge_reports(report, failure_report(f'{node_id}: metrics: {error}'))
            if original_error is None:
                original_error = error
        # report the run outcome even if shutdown fails
        try:
            if lifecycle is None:
                log('Shutdown coordination unavailable: control setup did not complete.')
                if original_error is not None:
                    raise original_error
            try:
                lifecycle.finish(report)
            except RunFailed as outcome:
                log(f'Run {"interrupted" if outcome.interrupted else "failed"}: {outcome}')
                log('Coordinated shutdown completed.')
                if original_error is not None and not isinstance(original_error, KeyboardInterrupt):
                    raise original_error
                raise outcome from None
            except (Exception, KeyboardInterrupt) as shutdown_error:
                if report:
                    outcome = RunFailed(report)
                    log(f'Run {"interrupted" if outcome.interrupted else "failed"}: {outcome}')
                log(f'Coordinated shutdown INCOMPLETE: {type(shutdown_error).__name__}: {shutdown_error}')
                if original_error is not None and not isinstance(original_error, KeyboardInterrupt):
                    raise original_error
                if isinstance(shutdown_error, KeyboardInterrupt):
                    raise RunFailed(failure_report('shutdown interrupted again by Ctrl+C', interrupted=True)) from None
                raise
            else:
                log('Run completed successfully. Coordinated shutdown completed.')
        finally:
            try:
                if server is not None and server.is_root and lifecycle is not None:
                    reports = lifecycle.work_reports
                    workers = set(topology['worker_specs'])
                    if workers <= reports.keys() and all(reports[w] is not None for w in workers):
                        contributions = sum(reports[w]['contributions'] for w in workers)
                        samples = sum(reports[w]['samples'] for w in workers)
                        partial = sum(reports[w]['partial_samples'] for w in workers)
                        if recorder:
                            recorder.record('accounting', server.applied_contributions, server.total_samples,
                                            contributions - server.applied_contributions, samples - server.total_samples, partial)
                        log(f'[TRAIN] applied_contributions={server.applied_contributions} applied_samples={server.total_samples} '
                            f'discarded_contributions={contributions - server.applied_contributions} '
                            f'discarded_samples={samples - server.total_samples} partial_discarded_samples={partial}')
                    else:
                        if recorder:
                            recorder.record('accounting', server.applied_contributions, server.total_samples, None, None, None)
                        log('[TRAIN] discarded work unknown: incomplete worker shutdown reports')
            finally:
                try:
                    try:
                        # save final counts and timing even with metrics disabled
                        if server is not None and server.is_root and (metrics_dir or recorder):
                            from pathlib import Path
                            result_dir = Path(metrics_dir) if metrics_dir else recorder.directory
                            result_dir.mkdir(parents=True, exist_ok=True)
                            with (result_dir / 'result.json').open('x') as result_file:
                                json.dump(dict(version=server.version, samples=server.total_samples,
                                               contributions=server.applied_contributions,
                                               elapsed=server.update_elapsed), result_file)
                    finally:
                        if recorder:
                            recorder.close()
                finally:
                    ctx.destroy(linger=0)
