"""Explicit process membership and endpoints; imports use only the standard library."""
import re
import math
from urllib.parse import urlsplit

_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z', re.ASCII)
_HOST = re.compile(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*\Z', re.ASCII)
_ROLES = {'root_ps', 'edge_ps', 'worker', 'eval'}
_DEFAULT_TRAINING = dict(learning_rate=0.01, batch_size=32, n_epochs=1, eval_interval=200, local_steps=1, seed=0, max_rounds=None, num_workers=2, cpu_threads=None, log_interval=10)


def _endpoint(value, label, *, bind=False):
    if not isinstance(value, str) or any(c.isspace() for c in value):
        raise ValueError(f'{label}: expected tcp://host:port')
    try:
        parsed = urlsplit(value)
        port = parsed.port
        host = parsed.hostname
    except ValueError as error:
        raise ValueError(f'{label}: invalid TCP endpoint {value!r}') from error
    if (parsed.scheme != 'tcp' or not host or port is None or not 1 <= port <= 65535
            or parsed.path or parsed.query or parsed.fragment or parsed.username is not None
            or parsed.password is not None or ':' in host
            or not (re.fullmatch(r'[A-Za-z0-9_.-]+', host, re.ASCII) or (bind and host == '*'))
            or (not bind and host in ('*', '0.0.0.0'))):
        raise ValueError(f'{label}: expected TCP hostname/IPv4 and port 1..65535; actual {value!r}')
    return port


def load_topology(path):
    import yaml
    with open(path) as file:
        return build_topology(yaml.safe_load(file))


def build_topology(raw):
    if not isinstance(raw, dict) or 'nodes' not in raw:
        raise ValueError('configuration requires a nodes list '
                         'with explicit id, role, host, parent and endpoint fields')
    if not isinstance(raw['nodes'], list):
        raise ValueError('nodes must be a list')
    mode = raw.get('mode', 'sync')
    if mode not in ('sync', 'async'):
        raise ValueError('mode must be sync or async')
    logging = raw.get('logging', {})
    if (not isinstance(logging, dict) or logging.keys() - {'trace_communication'}
            or type(logging.get('trace_communication', False)) is not bool):
        raise ValueError('logging.trace_communication must be a boolean')
    nodes = {}
    advertised = set()
    listeners = set()
    for entry in raw['nodes']:
        if not isinstance(entry, dict):
            raise ValueError('each node must be a mapping')
        cfg = dict(entry)
        node_id, role, host = cfg.get('id'), cfg.get('role'), cfg.get('host')
        if not isinstance(node_id, str) or not _ID.fullmatch(node_id):
            raise ValueError(f'invalid node id {node_id!r}; use 1..64 ASCII letters/digits/_.-')
        if node_id in nodes:
            raise ValueError(f'duplicate node id {node_id!r}')
        if role not in _ROLES:
            raise ValueError(f'{node_id}: invalid role {role!r}')
        if not isinstance(host, str) or not _HOST.fullmatch(host):
            raise ValueError(f'{node_id}: invalid deployment host {host!r}')
        if cfg.get('cpu') not in (None, 0) or cfg.get('bw') not in (None, 0, ''):
            raise ValueError(f'{node_id}: CPU/bandwidth throttling is temporarily unsupported; '
                             'remove cpu/bw limits for an unthrottled run')
        cfg['children'] = []
        if 'sync' in cfg or 'mode' in cfg:
            raise ValueError(f'{node_id}: use top-level mode to select sync or async training')
        if role in ('root_ps', 'edge_ps'):
            cfg['sync'] = mode == 'sync'
        if role != 'worker':
            if 'endpoint' not in cfg:
                raise ValueError(f'{node_id}: endpoint required')
            cfg.setdefault('bind_endpoint', cfg['endpoint'])
            endpoint_pairs = [('endpoint', 'bind_endpoint')]
            if role == 'root_ps':
                if 'stop_endpoint' not in cfg:
                    raise ValueError(f'{node_id}: stop_endpoint required')
                cfg.setdefault('stop_bind_endpoint', cfg['stop_endpoint'])
                endpoint_pairs.append(('stop_endpoint', 'stop_bind_endpoint'))
            for public, binding in endpoint_pairs:
                _endpoint(cfg[public], f'{node_id}.{public}')
                port = _endpoint(cfg[binding], f'{node_id}.{binding}', bind=True)
                if cfg[public] in advertised:
                    raise ValueError(f'duplicate advertised endpoint {cfg[public]!r}')
                advertised.add(cfg[public])
                listener = (host, port)
                if listener in listeners:
                    raise ValueError(f'conflicting listener on deployment host {host!r}, port {port}')
                listeners.add(listener)
        nodes[node_id] = cfg
    roots = [cfg for cfg in nodes.values() if cfg['role'] == 'root_ps']
    evals = [cfg for cfg in nodes.values() if cfg['role'] == 'eval']
    workers = {node_id: cfg for node_id, cfg in nodes.items() if cfg['role'] == 'worker'}
    if len(roots) != 1 or len(evals) != 1 or not workers:
        raise ValueError('topology requires exactly one root_ps, one eval and at least one worker')
    root, evaluator = roots[0], evals[0]
    for cfg in nodes.values():
        role, parent = cfg['role'], cfg.get('parent')
        if role in ('root_ps', 'eval'):
            if parent is not None:
                raise ValueError(f"{cfg['id']}: role {role} cannot have a parent")
            continue
        if parent not in nodes:
            raise ValueError(f"{cfg['id']}: unknown/missing parent {parent!r}")
        parent_cfg = nodes[parent]
        allowed = ('root_ps',) if role == 'edge_ps' else ('root_ps', 'edge_ps')
        if parent_cfg['role'] not in allowed:
            raise ValueError(f"{cfg['id']}: incompatible parent role {parent_cfg['role']}")
        parent_cfg['children'].append(cfg['id'])
        cfg['parent_endpoint'] = parent_cfg['endpoint']
    root_roles = {nodes[node_id]['role'] for node_id in root['children']}
    if root_roles == {'worker', 'edge_ps'}:
        raise ValueError('hybrid direct-worker/edge root topology is not supported')
    for cfg in nodes.values():
        if cfg['role'] in ('root_ps', 'edge_ps') and not cfg['children']:
            raise ValueError(f"{cfg['id']}: parameter server has no children")
    worker_ranks = {node_id: rank for rank, node_id in enumerate(sorted(workers))}
    for node_id, cfg in workers.items():
        if 'worker_rank' in cfg:
            raise ValueError('worker_rank is derived from sorted worker IDs, not configurable')
        cfg['worker_rank'] = worker_ranks[node_id]
    training = dict(_DEFAULT_TRAINING)
    training.update(raw.get('training', {}))
    for key in ('batch_size', 'n_epochs', 'eval_interval', 'local_steps', 'log_interval'):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f'training.{key} must be a positive integer')
    if type(training['seed']) is not int or not 0 <= training['seed'] < 2**32:
        raise ValueError('training.seed must be an integer in [0, 2**32)')
    lr = training['learning_rate']
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('training.learning_rate must be finite and positive')
    for key in ('max_rounds', 'cpu_threads'):
        value = training[key]
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f'training.{key} must be a positive integer or null')
    if type(training['num_workers']) is not int or training['num_workers'] < 0:
        raise ValueError('training.num_workers must be a nonnegative integer')
    workload = raw.get('workload', 'resnet_cifar10')
    if workload not in ('resnet_cifar10', 'lenet_mnist'):
        raise ValueError('workload must be resnet_cifar10 or lenet_mnist')
    monitoring = dict(enabled=True, interval_s=0.5, interface=None, console=False)
    supplied_monitoring = raw.get('monitoring', {})
    if not isinstance(supplied_monitoring, dict) or supplied_monitoring.keys() - monitoring.keys():
        raise ValueError('monitoring must be a mapping with enabled, interval_s, interface and/or console')
    monitoring.update(supplied_monitoring)
    if type(monitoring['enabled']) is not bool or type(monitoring['console']) is not bool:
        raise ValueError('monitoring.enabled and monitoring.console must be booleans')
    interval = monitoring['interval_s']
    if (isinstance(interval, bool) or not isinstance(interval, (int, float))
            or not math.isfinite(interval) or interval <= 0):
        raise ValueError('monitoring.interval_s must be finite and positive')
    interface = monitoring['interface']
    if interface is not None and (not isinstance(interface, str) or not interface.strip()):
        raise ValueError('monitoring.interface must be a nonempty interface name or null')
    timeouts = dict(startup_s=120.0, training_s=None, shutdown_s=10.0)
    supplied_timeouts = raw.get('timeouts', {})
    if not isinstance(supplied_timeouts, dict) or supplied_timeouts.keys() - timeouts.keys():
        raise ValueError('timeouts must contain only startup_s, training_s and/or shutdown_s')
    timeouts.update(supplied_timeouts)
    for name, seconds in timeouts.items():
        if name == 'training_s' and seconds is None:
            continue
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError(f'timeouts.{name} must be finite and positive (training_s may be null)')
    data = dict(root='/mnt/data', download=False, train_samples=None, eval_samples=None)
    supplied_data = raw.get('data', {})
    if not isinstance(supplied_data, dict) or supplied_data.keys() - data.keys():
        raise ValueError('data must contain only root, download, train_samples and/or eval_samples')
    data.update(supplied_data)
    if not isinstance(data['root'], str) or not data['root'].strip():
        raise ValueError('data.root must be a nonempty path string')
    if type(data['download']) is not bool:
        raise ValueError('data.download must be a boolean')
    for key in ('train_samples', 'eval_samples'):
        count = data[key]
        if count is not None and (type(count) is not int or count <= 0):
            raise ValueError(f'data.{key} must be a positive integer or null')
    metrics = dict(enabled=True, update_every=1, measure_overhead=True, trace_duration_s=0)
    supplied_metrics = raw.get('metrics', {})
    if not isinstance(supplied_metrics, dict) or supplied_metrics.keys() - metrics.keys():
        raise ValueError('metrics must contain only enabled, update_every, measure_overhead and/or trace_duration_s')
    metrics.update(supplied_metrics)
    if type(metrics['enabled']) is not bool:
        raise ValueError('metrics.enabled must be a boolean')
    if type(metrics['update_every']) is not int or metrics['update_every'] <= 0:
        raise ValueError('metrics.update_every must be a positive integer')
    if type(metrics['measure_overhead']) is not bool:
        raise ValueError('metrics.measure_overhead must be a boolean')
    duration = metrics['trace_duration_s']
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError('metrics.trace_duration_s must be finite and nonnegative')
    if duration and not metrics['enabled']:
        raise ValueError('tracing requires metrics.enabled')
    eval_cfg = raw.get('eval', {})
    return dict(metrics=metrics, mode=mode, trace_communication=logging.get('trace_communication', False), workload=workload, node_configs=nodes, root_id=root['id'], eval_id=evaluator['id'],
                eval_endpoint=evaluator['endpoint'], stop_endpoint=root['stop_endpoint'],
                worker_specs=workers, worker_ranks=worker_ranks, training=training, data=data, monitoring=monitoring, timeouts=timeouts,
                eval_accuracy_threshold=eval_cfg.get('accuracy_threshold', 0.90),
                eval_time_limit_s=eval_cfg.get('time_limit_s', 3600))
