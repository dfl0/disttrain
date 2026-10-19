"""Per-run setup/cleanup and telemetry; resource throttling is temporarily disabled."""
import time
import json
import shutil
import socket
import subprocess
import threading
import warnings
from dataclasses import dataclass, field

from protocol import Role, ParamSerializer
from topology import load_topology


def new_metrics_state():
    return {
        "sections": [],
        "n_parent_send_reqs": 0,
        "n_child_send_reqs": 0,
        "send_reqs_children": [],
    }


@dataclass
class RuntimeContext:
    hostname: str
    role: Role
    start_time: float
    metrics: dict = field(default_factory=new_metrics_state)
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


# ----  utility functions  ---------------------------------------------------+


def determine_role(node_id, topology):
    cfg = topology["node_configs"].get(node_id)
    if cfg is None:
        raise ValueError(f"Unknown node ID {node_id!r}; configured IDs: {list(topology['node_configs'])}")
    result = dict(cfg)
    result["role"] = Role[cfg["role"].upper()]
    return result


def collect_sys_metrics(stop_event, context, log, interval, interface=None):
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
    observe(lambda: psutil.cpu_percent(interval=None))  # Prime interval CPU measurement.
    while not stop_event.wait(interval):
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
                # Avoid repeatedly launching a failing command at high sample rates.
                throttle_command = None
        if stop_event.is_set():
            break
        log('[METRICS] ' + json.dumps(dict(
            t_s=now - context.start_time, interval_s=elapsed,
            sample_end_s=time.perf_counter() - context.start_time,
            host_cpu_percent=cpu,
            host_cpu_frequency_mhz=frequency.current if frequency else None,
            host_memory_percent=memory.percent if memory else None,
            process_rss_bytes=rss, host_temperatures=temperatures,
            host_throttled_now=throttled_now, host_throttled_since_boot=throttled_before,
            host_network_interface=interface if interface is not None else 'all',
            host_network_tx_bytes_per_s=tx, host_network_rx_bytes_per_s=rx)))


# ----  role functions  ------------------------------------------------------+


def main(config_file='config.yaml', *, node_id):
    topology = load_topology(config_file)
    role_cfg = determine_role(node_id, topology)
    role = role_cfg["role"]
    training = topology["training"]
    from transport import open_role_sockets, open_control_socket
    from lifecycle import Lifecycle, RunStopped, RunFailed, failure_report
    from ps import ParameterServer
    from models import (
        LeNet, LeNetCIFAR, AlexNetCIFAR, ResNet20CIFAR, BasicBlock,
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
    try:
        control = open_control_socket(ctx, role, role_cfg, topology, node_id)
        lifecycle = Lifecycle(control, node_id, topology["root_id"],
                              topology["node_configs"], topology["eval_id"], topology["timeouts"])
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
        if monitoring['enabled']:
            monitor_thread = threading.Thread(
                target=collect_sys_metrics, args=(monitor_stop, context, log),
                kwargs=dict(interval=monitoring['interval_s'], interface=monitoring['interface']),
                name=f'disttrain-metrics-{node_id}', daemon=True)
            monitor_thread.start()
        match role:
            case Role.ROOT_PS:
                ParameterServer(
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
                    global_state=context.metrics,
                    log=log,
                    node_id=node_id,
                    lifecycle=lifecycle,
                ).run()

            case Role.EDGE_PS:
                ParameterServer(
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
                    global_state=context.metrics,
                    log=log,
                    node_id=node_id,
                    lifecycle=lifecycle,
                ).run()

            case Role.WORKER:
                n_workers = len(topology["worker_specs"])
                run_worker(
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
        # Report run status independently of whether coordination succeeds.
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
            monitor_stop.set()
            try:
                if monitor_thread is not None and monitor_thread.ident is not None:
                    monitor_thread.join(timeout=2.0)
                    if monitor_thread.is_alive():
                        warnings.warn('metrics thread did not stop within 2 seconds', RuntimeWarning)
            finally:
                # All training sockets belong to this thread/context. Destroy also
                # closes sockets left by partial setup or exceptions in any role.
                # Zero linger preserves current best-effort shutdown, not delivery ACKs.
                ctx.destroy(linger=0)
