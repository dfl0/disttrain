"""Run nodes locally or over SSH and collect their logs."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid

SSH_OPTIONS = ['-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2']
EXIT_MARKER = '[DISTTRAIN_LAUNCHER_EXIT] '


def launch_entries(topology):
    nodes = topology['node_configs']
    ordered = [topology['root_id']]
    ordered.extend(node for node, cfg in nodes.items() if cfg['role'] == 'edge_ps')
    ordered.append(topology['eval_id'])
    ordered.extend(node for node, cfg in nodes.items() if cfg['role'] == 'worker')
    return [(nodes[node]['host'], node, nodes[node]['role']) for node in ordered]


def signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can report EPERM after the process group exits
        if process.poll() is None:
            raise


def remote_node():
    """Run one remote node. Stop it if the SSH input closes.

    Stdin accepts a JSON launch request, then INT/TERM/KILL commands.
    """
    startup = bytearray()
    while not startup.endswith(b'\n'):
        chunk = os.read(sys.stdin.fileno(), 1)
        if not chunk:
            raise ValueError('SSH closed before startup request')
        startup.extend(chunk)
    request = json.loads(startup)
    process = None
    result = dict(token=request['token'], exit_code=None, cleanup_confirmed=False)
    interrupted = False

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    try:
        # keep metrics available for copying after the node exits
        metrics_dir = str(Path(tempfile.mkdtemp(prefix='disttrain-metrics-')) / 'metrics')
        result['metrics_dir'] = metrics_dir
        with tempfile.TemporaryDirectory(prefix='disttrain-') as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text(request['config'])
            process = subprocess.Popen(
                [request['python'], '-u', 'dist_train.py', '--config', str(config),
                 '--node-id', request['node_id'], '--metrics-dir', metrics_dir], stdin=subprocess.DEVNULL,
                stderr=subprocess.STDOUT, start_new_session=True)
            with selectors.DefaultSelector() as selector:
                selector.register(sys.stdin.buffer, selectors.EVENT_READ)
                # buffered reads can hide pending commands from the selector
                pending = b''
                deadline = None
                phase = None
                while process.poll() is None:
                    commands = []
                    for key, _ in selector.select(timeout=0.1):
                        chunk = os.read(key.fd, 4096)
                        if not chunk:
                            selector.unregister(key.fd)
                            interrupted = True
                        else:
                            pending += chunk
                            while b'\n' in pending:
                                line, pending = pending.split(b'\n', 1)
                                commands.append(line.decode())
                    if process.poll() is not None:
                        break
                    if interrupted and phase is None:
                        commands.append('INT')
                    for command in commands:
                        if command not in ('INT', 'TERM', 'KILL'):
                            raise ValueError('invalid supervisor command')
                        stage = ('INT', 'TERM', 'KILL').index(command)
                        if phase is None or stage > phase:
                            signal_group(process, (signal.SIGINT, signal.SIGTERM, signal.SIGKILL)[stage])
                            phase = stage
                            deadline = time.monotonic() + (request['shutdown_s'] if stage == 0 else 2)
                    if deadline is not None and time.monotonic() >= deadline:
                        if phase == 0:
                            signal_group(process, signal.SIGTERM)
                            phase, deadline = 1, time.monotonic() + 2
                        elif phase == 1:
                            signal_group(process, signal.SIGKILL)
                            phase, deadline = 2, time.monotonic() + 2
                        else:
                            raise TimeoutError('remote child did not exit after SIGKILL')
            result['exit_code'] = process.returncode
            # stop any remaining DataLoader processes
            signal_group(process, signal.SIGKILL)
            result['cleanup_confirmed'] = True
    except Exception as error:
        import traceback
        traceback.print_exc()
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        if process is not None and process.poll() is None:
            signal_group(process, signal.SIGKILL)
            try:
                result['exit_code'] = process.wait(timeout=2)
                result['cleanup_confirmed'] = True
            except subprocess.TimeoutExpired:
                pass
        print('\n' + EXIT_MARKER + json.dumps(result), flush=True)
    return 0 if result['cleanup_confirmed'] else 1


def remote_result(path, token):
    with path.open('rb') as file:
        file.seek(0, 2)
        file.seek(max(0, file.tell() - 65536))
        lines = file.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        if line.startswith(EXIT_MARKER):
            try:
                result = json.loads(line[len(EXIT_MARKER):])
                if result.get('token') == token:
                    return result
            except (ValueError, AttributeError):
                pass
    return dict(exit_code=None, cleanup_confirmed=False,
                error='SSH ended without a remote exit report; remote cleanup unconfirmed')


def collect_metrics(output, entries, nodes, *, mode, enabled=True):
    for host, node, role in entries:
        target = output / f'{node}.metrics'
        status = nodes[node]
        status['metrics'] = target.name
        if not enabled:
            if role == 'root_ps':
                if mode == 'ssh' and status.get('metrics_dir'):
                    target.mkdir(exist_ok=True)
                    try:
                        result = subprocess.run(
                            ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             f'{host}:{status["metrics_dir"]}/result.json', str(target / 'result.json')],
                            capture_output=True, text=True, timeout=30)
                        if result.returncode:
                            status['result_error'] = result.stderr.strip() or f'scp exited {result.returncode}'
                    except (OSError, subprocess.TimeoutExpired) as error:
                        status['result_error'] = str(error)
                status['result_status'] = 'complete' if (target / 'result.json').is_file() else 'missing'
            status['metrics_files'] = []
            status['metrics_status'] = 'disabled'
            continue
        if mode == 'local':
            files = sorted(path.name for path in target.glob('*.csv')) if target.is_dir() else []
        else:
            remote_dir = status.get('metrics_dir')
            files = []
            if remote_dir:
                target.mkdir(exist_ok=True)
                # remote_dir comes from mkdtemp on the remote host
                try:
                    result = subprocess.run(
                        ['scp', '-q', '-r', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                         f'{host}:{remote_dir}/.', str(target)],
                        capture_output=True, text=True, timeout=30)
                    if result.returncode:
                        status['metrics_error'] = result.stderr.strip() or f'scp exited {result.returncode}'
                    else:
                        files = sorted(path.name for path in target.glob('*.csv'))
                except (OSError, subprocess.TimeoutExpired) as error:
                    status['metrics_error'] = f'{type(error).__name__}: {error}'
        status['metrics_files'] = files
        if status.get('metrics_error'):
            status['metrics_status'] = 'failed'
        elif 'recording.csv' not in files:
            status['metrics_status'] = 'missing' if not files else 'incomplete'
        else:
            try:
                with (target / 'recording.csv').open(newline='', encoding='utf-8') as file:
                    records = list(csv.DictReader(file))
                status['metrics_status'] = ('complete' if len(records) == 1 and
                                            records[0].get('complete') == 'True' else 'incomplete')
            except (OSError, UnicodeError, csv.Error) as error:
                status['metrics_status'] = 'incomplete'
                status['metrics_error'] = f'{type(error).__name__}: {error}'


def run(args, topology, config_text):
    output = Path(args.output_dir) / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'config.yaml'
    config_path.write_text(config_text)
    entries = launch_entries(topology)
    nodes = {node: dict(host=host, role=role, exit_code=None, cleanup_confirmed=False,
                        log=f'{node}.log', state='not_started') for host, node, role in entries}
    processes = {}
    files = {}
    tokens = {}
    started = time.time()
    stop_requested = False
    interrupted = False
    failure = None
    stop_phase = None
    deadline = None

    def on_signal(signum, frame):
        nonlocal stop_requested, interrupted
        stop_requested = interrupted = True

    def send(node, sig):
        process = processes[node]
        if args.mode == 'local':
            signal_group(process, sig)
        elif process.poll() is None:
            try:
                process.stdin.write({signal.SIGINT: b'INT\n', signal.SIGTERM: b'TERM\n',
                                     signal.SIGKILL: b'KILL\n'}[sig])
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass  # check the exit report when SSH closes

    previous_handlers = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        try:
            for host, node, role in entries:
                if stop_requested:
                    break
                log_path = output / nodes[node]['log']
                files[node] = log_path.open('wb')
                if args.mode == 'local':
                    command = [args.python, '-u', 'dist_train.py', '--config', str(config_path.resolve()),
                               '--node-id', node, '--metrics-dir', str((output / f'{node}.metrics').resolve())]
                else:
                    tokens[node] = uuid.uuid4().hex
                    remote = f'cd {shlex.quote(args.workdir)} && exec {shlex.join([args.python, "-u", "launcher.py", "--remote-node"])}'
                    command = ['ssh', *SSH_OPTIONS, host, remote]
                process = subprocess.Popen(command, cwd=args.workdir if args.mode == 'local' else None,
                                           stdin=subprocess.PIPE if args.mode == 'ssh' else subprocess.DEVNULL,
                                           stdout=files[node], stderr=subprocess.STDOUT, start_new_session=True)
                processes[node] = process
                nodes[node]['state'] = 'running'
                if args.mode == 'ssh':
                    request = dict(token=tokens[node], config=config_text, node_id=node,
                                   python=args.python, shutdown_s=topology['timeouts']['shutdown_s'])
                    process.stdin.write(json.dumps(request).encode() + b'\n')
                    process.stdin.flush()
                print(f'Started {node} ({role}) on {host if args.mode == "ssh" else "localhost"}', flush=True)
        except Exception as error:
            failure = f'launch failed: {type(error).__name__}: {error}'
            stop_requested = True

        while any(node['state'] == 'running' for node in nodes.values()):
            for node, process in processes.items():
                if nodes[node]['state'] != 'running' or process.poll() is None:
                    continue
                files[node].close()
                if args.mode == 'local':
                    signal_group(process, signal.SIGKILL)
                    result = dict(exit_code=process.returncode, cleanup_confirmed=True)
                else:
                    result = remote_result(output / nodes[node]['log'], tokens[node])
                    result.pop('token', None)
                    result['ssh_exit_code'] = process.returncode
                    process.stdin.close()
                nodes[node].update(result, state='exited')
                print(f'Exited {node}: code={result["exit_code"]}, cleanup={"confirmed" if result["cleanup_confirmed"] else "UNCONFIRMED"}', flush=True)
                if result['exit_code'] != 0 or not result['cleanup_confirmed'] or result.get('error') or result.get('ssh_exit_code', 0) != 0:
                    failure = failure or f'{node} exited unsuccessfully'
                    stop_requested = True
            active = [node for node in processes if nodes[node]['state'] == 'running']
            if not active:
                break
            if stop_requested and stop_phase is None:
                stop_phase = 0
                deadline = time.monotonic() + topology['timeouts']['shutdown_s']
                print('Stopping run: allowing coordinated shutdown...', flush=True)
                # a failed node may already have requested shutdown
                # another SIGINT would interrupt that cleanup
                if interrupted or failure and failure.startswith('launch failed:'):
                    target = topology['root_id'] if topology['root_id'] in active else active[0]
                    send(target, signal.SIGINT)
            elif deadline is not None and time.monotonic() >= deadline:
                if stop_phase < 2:
                    stop_phase += 1
                    sig = signal.SIGTERM if stop_phase == 1 else signal.SIGKILL
                    print(f'Escalating to {sig.name} for remaining owned processes', flush=True)
                    for node in active:
                        send(node, sig)
                    deadline = time.monotonic() + 2
                else:
                    # closing SSH triggers remote cleanup, but can't confirm it succeeded
                    for node in active:
                        signal_group(processes[node], signal.SIGKILL)
                        try:
                            processes[node].wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                        nodes[node].update(state='unconfirmed', cleanup_confirmed=False,
                                           error='no exit confirmation after forced shutdown')
                    failure = failure or 'shutdown could not be confirmed'
                    break
            time.sleep(0.1)  # poll process exits, not training messages
    except Exception as error:
        failure = f'launcher error: {type(error).__name__}: {error}'
    finally:
        for node, process in processes.items():
            if process.poll() is None:
                if args.mode == 'ssh':
                    send(node, signal.SIGKILL)
                    process.stdin.close()  # closing stdin also requests remote cleanup
                signal_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                nodes[node].update(state='unconfirmed', cleanup_confirmed=False)
                failure = failure or 'cleanup unconfirmed'
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        for file in files.values():
            file.close()
        collect_metrics(output, entries, nodes, mode=args.mode,
                        enabled=topology.get('metrics', {}).get('enabled', True))
        for node, status in nodes.items():
            if status['exit_code'] == 0 and status['metrics_status'] not in ('complete', 'disabled'):
                failure = failure or f'{node} metrics collection {status["metrics_status"]}'
        outcome = 'interrupted' if interrupted else 'failed' if failure else 'success'
        summary = dict(outcome=outcome, error=failure, mode=args.mode, started_at=started,
                       finished_at=time.time(), workdir=args.workdir, python=args.python, nodes=nodes)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    print(f'Run {outcome}. Logs and summary: {output}', flush=True)
    return 130 if interrupted else 1 if failure else 0


def main(argv=None):
    if argv is None and sys.argv[1:] == ['--remote-node']:
        return remote_node()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('run_name', nargs='?', help='output directory name; defaults to the config filename without its extension')
    parser.add_argument('--mode', choices=('local', 'ssh'), required=True)
    parser.add_argument('--workdir', help='code directory (required for SSH); data paths are relative to it')
    parser.add_argument('--python', help='Python executable on target; default: current Python locally, venv/bin/python remotely')
    parser.add_argument('--output-dir', default='logs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.run_name is None:
        args.run_name = Path(args.config).stem
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name) or args.run_name in ('.', '..'):
        parser.error('run_name must be a simple filename using letters, digits, underscore, dot or hyphen')
    if args.mode == 'ssh' and not args.workdir:
        parser.error('--workdir is required for SSH mode')
    args.workdir = args.workdir or str(Path.cwd())
    args.python = args.python or (sys.executable if args.mode == 'local' else 'venv/bin/python')
    try:
        config_text = Path(args.config).read_text()
        # validate the config that will be sent to every node
        import yaml
        from topology import build_topology
        topology = build_topology(yaml.safe_load(config_text))
        if args.dry_run:
            for host, node, role in launch_entries(topology):
                print(f'{node} ({role}) on {host if args.mode == "ssh" else "localhost"}: '
                      f'{args.python} in {args.workdir}')
            return 0
        return run(args, topology, config_text)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
