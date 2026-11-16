"""Generate an offline HTML comparison from saved run directories."""
import argparse
import csv
import html
import json
import math
import copy
import statistics
from pathlib import Path

import yaml
from clock_alignment import ClockAlignment


def server_policies(config):
    return {node['id']: node.get('mode', config.get('mode', 'sync'))
            for node in config['nodes'] if node['role'] in ('root_ps', 'edge_ps')}


def policy_label(config):
    policies = server_policies(config)
    root = next(node for node in config['nodes'] if node['role'] == 'root_ps')
    edges = [node for node in config['nodes'] if node['role'] == 'edge_ps']
    if not edges:
        return f"{policies[root['id']]} / flat"
    edge_modes = {policies[node['id']] for node in edges}
    label = next(iter(edge_modes)) if len(edge_modes) == 1 else ', '.join(
        f"{node['id']}={policies[node['id']]}" for node in edges)
    return f"root {policies[root['id']]} / edges {label} / hierarchical"


def read_run(directory):
    from metrics import SCHEMAS
    directory = Path(directory)
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    if not isinstance(config, dict) or not isinstance(config.get('nodes'), list):
        raise ValueError(f'{directory}: config requires a nodes list')
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    run = dict(name=directory.name, path=str(directory.resolve()), config=config,
               summary=summary, progress=[], evaluations=[], resources=[], accounting={}, overhead=[], recordings={}, recording_windows={}, final_result={}, warnings=[])
    integers = {'version', 'samples', 'contributions', 'aggregate_samples', 'aggregate_contributions',
                'staleness_weighted_sum', 'staleness_max', 'eval_samples', 'process_rss_bytes',
                'applied_contributions', 'applied_samples', 'discarded_contributions', 'discarded_samples',
                'partial_discarded_samples', 'records', 'queue_full_events', 'calls'}
    text_fields = {'component', 'host_network_interface', 'group', 'label', 'complete',
                   'host_throttled_now', 'host_throttled_since_boot'}
    def read_csv(path, stream):
        if not path.exists():
            return []
        with path.open(newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != list(SCHEMAS[stream]):
                raise ValueError(f'{path}: unexpected CSV columns')
            records = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'{path}: incomplete CSV record')
                records.append({key: None if value == '' else value if key in text_fields
                                else int(value) if key in integers else float(value)
                                for key, value in row.items()})
            return records
    clocks = ClockAlignment(directory, {node['id']: node for node in config['nodes']},
                            local=summary.get('mode') == 'local')
    run['alignment'] = [clocks.describe(node['id']) for node in config['nodes']]
    found = False
    for node in config['nodes']:
        node_id, role = node['id'], node['role']
        folder = directory / f'{node_id}.metrics'
        if not folder.exists():
            if config.get('metrics', {}).get('enabled', True):
                run['warnings'].append(f'{node_id}: no CSV measurements (disabled, missing, or old run).')
            continue
        found = True
        if role == 'root_ps' and (folder / 'result.json').exists():
            run['final_result'] = json.loads((folder / 'result.json').read_text())
        window = read_csv(folder / 'recording_window.csv', 'recording_window')
        if window:
            run['recording_windows'][node_id] = window[-1]['elapsed_s']
        if not config.get('metrics', {}).get('enabled', True):
            continue
        run['overhead'].extend(dict(node=node_id, **point) for point in read_csv(folder / 'overhead.csv', 'overhead'))
        recording = read_csv(folder / 'recording.csv', 'recording')
        if recording:
            run['recordings'][node_id] = recording[-1]
        if not recording or recording[-1]['complete'] != 'True':
            run['warnings'].append(f'{node_id}: CSV recording did not finish cleanly; data may be incomplete.')
        elif recording[-1]['queue_full_events']:
            run['warnings'].append(f'{node_id}: recorder queue filled {recording[-1]["queue_full_events"]} times; '
                                   f'producers waited {recording[-1]["queue_wait_s"]:.6f}s.')
        if role == 'root_ps':
            run['progress'] = read_csv(folder / 'updates.csv', 'updates')
            for point in run['progress']:
                point['staleness_mean'] = point['staleness_weighted_sum'] / point['samples'] if point['samples'] else 0
                if all(mode == 'sync' for mode in server_policies(config).values()):
                    point.pop('staleness_mean')
                    point.pop('staleness_max')
            accounting = read_csv(folder / 'accounting.csv', 'accounting')
            run['accounting'] = accounting[-1] if accounting else {}
        elif role == 'eval':
            run['evaluations'] = read_csv(folder / 'evaluations.csv', 'evaluations')
            for point in run['evaluations']:
                received = clocks.elapsed(node_id, 'evaluation', point['received'])
                done = clocks.elapsed(node_id, 'evaluation', point['done'])
                if received is not None and done is not None:
                    point['received_root_s'], point['done_root_s'] = received[0], done[0]
                    point['result_delay_s'] = done[0] - point['root_elapsed']
                    point['snapshot_wait_s'] = received[0] - point['root_elapsed']
                    point['alignment_error_s'] = max(received[1], done[1])
                    if received[2] or done[2]:
                        run['warnings'].append(f'{node_id}: evaluation outside calibration interval; drift uncertainty unknown.')
        resources = read_csv(folder / 'resources.csv', 'resources')
        outside = False
        for point in resources:
            aligned = clocks.elapsed(node_id, 'runtime', point['t_s'])
            if aligned is not None:
                point['aligned_s'], point['alignment_error_s'], extrapolated = aligned
                outside |= extrapolated
            run['resources'].append(dict(node=node_id, **point))
        if resources and not clocks.elapsed(node_id, 'runtime', resources[0]['t_s']):
            run['warnings'].append(f'{node_id}: no clock mapping; resource plots use node-local time separately.')
        if outside:
            run['warnings'].append(f'{node_id}: resource samples outside calibration interval use nearest offset; drift uncertainty unknown.')
    if not found and config.get('metrics', {}).get('enabled', True):
        raise ValueError(f'{directory}: no CSV measurements; rerun with metrics enabled (console-log parsing removed)')
    if not summary:
        run['warnings'].append('No launcher summary: completion status unknown.')
    if summary and summary.get('outcome') != 'success':
        run['warnings'].append(f'Launcher outcome: {summary.get("outcome", "unknown")}')
    for node, status in summary.get('nodes', {}).items():
        if status.get('exit_code') != 0 or not status.get('cleanup_confirmed'):
            run['warnings'].append(f'{node}: nonzero/missing exit code or unconfirmed cleanup.')
        if status.get('metrics_status') in ('missing', 'failed'):
            run['warnings'].append(f'{node}: metric collection {status["metrics_status"]}')
    if run['progress'] and run['evaluations'] and run['progress'][-1]['version'] != run['evaluations'][-1]['version']:
        run['warnings'].append('Last evaluated version differs from last recorded root version.')
    if not run['progress'] and not run['final_result']:
        run['warnings'].append('No root update measurements.')
    if not run['evaluations'] and config.get('metrics', {}).get('enabled', True):
        run['warnings'].append('No evaluation measurements.')
    for before, after in zip(run['progress'], run['progress'][1:]):
        if after['version'] <= before['version'] or after['samples'] <= before['samples'] or after['elapsed'] < before['elapsed']:
            run['warnings'].append('Non-monotonic root measurements; inspect this run before comparing.')
            break
    return run


def interval_points(progress, interval_s=0.1):
    """Group precise update records into display intervals without changing collection."""
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("display interval must be finite and positive")
    previous = dict(elapsed=0, samples=0, version=0, contributions=0)
    points = []
    last_seen = previous
    for index, current in enumerate(progress):
        if (current["version"] <= last_seen["version"] or current["samples"] < last_seen["samples"]
                or current["elapsed"] < last_seen["elapsed"]):
            points.append(dict(elapsed=current["elapsed"], throughput=None, aggregate=None))
            previous = last_seen = current
            continue
        last_seen = current
        dt = current['elapsed'] - previous['elapsed']
        if dt < interval_s and index != len(progress) - 1:
            continue
        dv = current['version'] - previous['version']
        ds = current['samples'] - previous['samples']
        point = dict(elapsed=current['elapsed'], throughput=None, aggregate=None)
        if dt > 0 and ds >= 0 and dv > 0:
            point['throughput'] = ds / dt
        if dv > 0 and 'contributions' in current and 'contributions' in previous:
            dc = current['contributions'] - previous['contributions']
            if dc >= 0:
                point['aggregate'] = dc / dv
        points.append(point)
        previous = current
    return points


def compare_instrumentation(runs, baseline_paths):
    """Compare repeated runs with matching work and configuration."""
    if not baseline_paths:
        return []
    baselines = {str(Path(path).resolve()) for path in baseline_paths}
    if not baselines <= {run['path'] for run in runs}:
        raise ValueError('every --baseline must also be included among the input run directories')

    def signatures(run):
        config = copy.deepcopy(run['config'])
        metrics = dict(enabled=True, update_every=1, measure_overhead=True, trace_duration_s=0)
        metrics.update(config.pop('metrics', {}))
        monitoring = dict(enabled=True, interval_s=.5, interface=None, console=False)
        monitoring.update(config.pop('monitoring', {}))
        config['nodes'] = sorted(config['nodes'], key=lambda node: node['id'])
        # match launch settings as well as the training configuration
        config['launch_mode'] = run['summary'].get('mode')
        config['launch_workdir'] = run['summary'].get('workdir')
        config['launch_python'] = run['summary'].get('python')
        return json.dumps(config, sort_keys=True), json.dumps([metrics, monitoring], sort_keys=True)

    reference = next(run for run in runs if run['path'] in baselines)
    workload, reference_settings = signatures(reference)
    groups = {}
    sample_budget = None
    for run in runs:
        match, settings = signatures(run)
        if match != workload:
            raise ValueError(f'{run["name"]}: baseline comparison requires matching config/launch settings except metrics and monitoring')
        if run['summary'].get('outcome') != 'success':
            raise ValueError(f'{run["name"]}: baseline comparison requires successful runs')
        if run['path'] in baselines and settings != reference_settings:
            raise ValueError('baseline repeats must use identical instrumentation settings')
        result = run['final_result'] or (run['progress'][-1] if run['progress'] else {})
        samples, elapsed = result.get('samples'), result.get('elapsed')
        if type(samples) is not int or samples <= 0 or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f'{run["name"]}: exact final work/timing unavailable; rerun with result.json support')
        if not all(key in run['summary'] for key in ('started_at', 'finished_at')):
            raise ValueError(f'{run["name"]}: launcher timestamps unavailable')
        wall = run['summary']['finished_at'] - run['summary']['started_at']
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError(f'{run["name"]}: invalid launcher duration')
        if sample_budget is None:
            sample_budget = samples
        if samples != sample_budget:
            raise ValueError('baseline comparison requires equal applied-sample counts; update counts alone are insufficient')
        is_baseline = run['path'] in baselines
        group = groups.setdefault((is_baseline, settings), dict(baseline=is_baseline, settings=settings,
                                                               training=[], launcher=[], names=[]))
        group['training'].append(elapsed)
        group['launcher'].append(wall)
        group['names'].append(run['name'])
    reference_group = next(group for group in groups.values() if group['baseline'])
    result_rows = []
    for group in sorted(groups.values(), key=lambda group: not group['baseline']):
        metrics, monitoring = json.loads(group['settings'])
        monitoring_active = monitoring['enabled'] and (metrics['enabled'] or monitoring['console'])
        label = ('Baseline: ' if group['baseline'] else '') + (
            f'CSV {"on" if metrics["enabled"] else "off"}; resources {"on" if monitoring_active else "off"}; '
            f'trace {metrics["trace_duration_s"]:g}s')
        row = dict(label=label, count=len(group['names']), names=group['names'], baseline=group['baseline'],
                   samples=sample_budget, settings=group['settings'])
        for key in ('training', 'launcher'):
            values = group[key]
            reference_median = statistics.median(reference_group[key])
            median = statistics.median(values)
            row[key] = dict(median=median, low=min(values), high=max(values),
                            slowdown=100 * (median / reference_median - 1),
                            low_pct=100 * (min(values) / reference_median - 1),
                            high_pct=100 * (max(values) / reference_median - 1))
        result_rows.append(row)
    return result_rows


def render_report(runs, display_interval_s=0.1, baseline_paths=()):
    if not math.isfinite(display_interval_s) or display_interval_s <= 0:
        raise ValueError('display interval must be finite and positive')
    import plotly.graph_objects as go
    from plotly.io import to_html
    from plotly.offline import get_plotlyjs
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    escape = lambda value: html.escape(str(value), quote=True)
    sections = []
    chart_count = 0
    import plotly.express as px
    colors = px.colors.qualitative.T10
    role_dashes = {'root_ps': 'solid', 'edge_ps': 'dash', 'eval': 'dot', 'worker': 'dashdot'}
    device_dashes = {(index, node['id']): role_dashes[node['role']]
                     for index, run in enumerate(runs) for node in run['config']['nodes']}

    def chart(title, x_title, y_title, series, note='', group_runs=False, secondary_y=None):
        nonlocal chart_count
        figure = go.Figure()
        groups = {}
        for index, name, points, hover in series:
            valid = [(x, y, detail) for x, y, detail in zip(points[0], points[1], hover)
                     if isinstance(x, (int, float)) and isinstance(y, (int, float))
                     and math.isfinite(x) and math.isfinite(y)]
            if not valid:
                continue
            if group_runs:
                groups.setdefault(index, dict(name=runs[index]['name'], indices=[]))['indices'].append(len(figure.data))
            xs, ys, details = zip(*valid)
            node = name.removeprefix(runs[index]['name'] + ' / ')
            dash = device_dashes.get((index, node), 'solid') if group_runs else 'solid'
            figure.add_trace(go.Scatter(x=xs, y=ys, text=details, name=escape(name),
                                       mode='lines', yaxis='y2' if secondary_y and node != 'throughput' else 'y',
                                       line=dict(color='#111827' if secondary_y and node == 'throughput' else colors[index % len(colors)],
                                                 width=3 if secondary_y and node == 'throughput' else 2, dash=dash),
                                       hovertemplate='%{text}<br>x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>'))
        if not figure.data:
            return
        # reserve legend space so toggling traces does not move the plot
        figure.update_layout(xaxis_title=x_title, yaxis_title=y_title,
                             template='plotly_white', height=480,
                             margin=dict(t=30, b=80, l=75, r=390, autoexpand=False),
                             legend=dict(orientation='v', x=1.02, xanchor='left',
                                         y=1, yanchor='top', font=dict(size=12)),
                             xaxis=dict(automargin=True, title_standoff=18),
                             yaxis=dict(automargin=True))
        if secondary_y:
            figure.update_layout(yaxis2=dict(title=secondary_y, overlaying='y', side='right', range=[0, 100]))
        # embed Plotly once in the template head
        fragment = to_html(figure, full_html=False, include_plotlyjs=False,
                           config=dict(displaylogo=False, responsive=True), div_id=f'chart-{chart_count}')
        sections.append(dict(title=title, html=fragment, note=note,
                             id=f"chart-{chart_count}", groups=list(groups.values())))
        chart_count += 1

    def series_for(source, x, y):
        return [(i, run['name'], ([point.get(x) for point in run[source]],
                                  [point.get(y) for point in run[source]]),
                 [f'version={point.get("version", "—")}; samples={point.get("samples", "—")}; alignment error estimate (s)={point.get("alignment_error_s", "—")}'
                  for point in run[source]]) for i, run in enumerate(runs)]

    rows = []
    for run in runs:
        config, summary = run['config'], run['summary']
        final = run['final_result'] or (run['progress'][-1] if run['progress'] else {})
        evaluation = run['evaluations'][-1] if run['evaluations'] else {}
        accounting = run['accounting']
        wall = (summary['finished_at'] - summary['started_at']
                if all(key in summary for key in ('started_at', 'finished_at')) else None)
        fmt = lambda value, suffix='': '—' if value is None else f'{value}{suffix}'
        rows.append(dict(
            name=run['name'],
            mode=policy_label(config),
            workers=sum(n['role'] == 'worker' for n in config['nodes']),
            status=summary.get('outcome', 'unknown'), updates=final.get('version', '—'),
            contributions=accounting.get('applied_contributions',
                int(final['contributions']) if 'contributions' in final else '—'),
            samples=final.get('samples', '—'),
            discarded=f'{fmt(accounting.get("discarded_contributions"))} / '
                      f'{fmt(accounting.get("discarded_samples"))} / '
                      f'{fmt(accounting.get("partial_discarded_samples"))}',
            evaluated_version=evaluation.get('version', '—'),
            accuracy=fmt(evaluation.get('accuracy'), '%'), loss=fmt(evaluation.get('loss')),
            training_elapsed=fmt(final.get('elapsed'), ' s'),
            launcher_elapsed=fmt(round(wall, 3) if wall is not None else None, ' s')))
    settings = []
    for run in runs:
        cfg = run['config']
        training, data = cfg.get('training', {}), cfg.get('data', {})
        workers = sum(n['role'] == 'worker' for n in cfg['nodes'])
        settings.append([cfg.get('workload', 'resnet_cifar10'), training.get('learning_rate', .01),
                         training.get('batch_size', 32), training.get('local_steps', 1),
                         workers * training.get('batch_size', 32) * training.get('local_steps', 1)
                         if 'async' in server_policies(cfg).values() else 'actual round samples',
                         training.get('seed', 0), training.get('n_epochs', 1),
                         training.get('max_rounds') or 'none', data.get('train_samples') or 'full split',
                         data.get('eval_samples') or 'full split', training.get('eval_interval', 200),
                         cfg.get('metrics', {}).get('update_every', 1),
                         cfg.get('metrics', {}).get('measure_overhead', True),
                         cfg.get('metrics', {}).get('trace_duration_s', 0)])
    labels = ['Workload', 'Learning rate', 'Batch size', 'Local steps', 'Normalization denominator',
              'Seed', 'Epochs', 'Update cap', 'Training subset', 'Evaluation subset', 'Evaluation interval (updates)', 'Record every N updates', 'Measure recording overhead', 'Trace capture window (s)']
    setting_rows = []
    for column, label in enumerate(labels):
        values = [row[column] for row in settings]
        setting_rows.append(dict(label=label, cells=values,
                                 different=len({str(v) for v in values}) > 1))
    chart('Applied training samples', 'Root training elapsed (s)', 'Applied samples',
          series_for('progress', 'elapsed', 'samples'))
    for metric, title in [('accuracy', 'Accuracy (%)'), ('loss', 'Loss')]:
        chart(f'{title} versus work applied', 'Applied samples at evaluated version', title,
              series_for('evaluations', 'samples', metric), 'Sample counts travel with each evaluated snapshot; no console-log matching or interpolation.')
        chart(f'{title} versus model availability', 'Root training elapsed at model update (s)', title,
              series_for('evaluations', 'root_elapsed', metric),
              'Root timestamp carried with the snapshot; does not include subsequent evaluation time.')
        chart(f'{title} versus evaluation completion time', 'Root training elapsed (s)', title,
              series_for('evaluations', 'done_root_s', metric),
              'Completion mapped to root time; calibration uncertainty applies. Uncalibrated runs omitted.')
    chart('Evaluation result delay', 'Root elapsed at model availability (s)', 'Seconds until evaluation completes',
          series_for('evaluations', 'root_elapsed', 'result_delay_s'),
          'Includes snapshot delivery/queueing, evaluator scheduling and evaluation. Not network latency. Small negative values can reflect alignment error.')
    chart('Snapshot availability to evaluation start', 'Root elapsed at model availability (s)', 'Delay (s)',
          series_for('evaluations', 'root_elapsed', 'snapshot_wait_s'),
          'Time until evaluator drains/selects the snapshot, before loading/evaluating; includes queueing and scheduling, not just transport.')
    for run in runs:
        run['intervals'] = interval_points(run['progress'], display_interval_s)
    chart(f'Training throughput ({display_interval_s:g} s display intervals)', 'Root training elapsed (s)', 'Applied samples / s',
          series_for('intervals', 'elapsed', 'throughput'),
          f'Rates group available CSV records into approximately {display_interval_s:g} s intervals (last may be shorter). Sparse recording limits resolution; no missing updates are reconstructed.')
    chart(f'Aggregate size ({display_interval_s:g} s display intervals)', 'Root training elapsed (s)', 'Contributions / root update',
          series_for('intervals', 'elapsed', 'aggregate'),
          'Computed from exact contribution/update deltas over display intervals; raw update records remain in CSV.')
    for key, label in [('staleness_mean', 'Sample-weighted cumulative mean staleness'),
                       ('staleness_max', 'Cumulative maximum staleness')]:
        chart(label, 'Root training elapsed (s)', 'Root updates behind', series_for('progress', 'elapsed', key))
    resources = [('host_cpu_percent', 'Host CPU utilization', '%', 1),
                 ('host_memory_percent', 'Host memory utilization', '%', 1),
                 ('process_rss_bytes', 'Process resident memory', 'MiB', 1 / 2**20),
                 ('host_network_tx_bytes_per_s', 'Host network transmit', 'MiB/s', 1 / 2**20),
                 ('host_network_rx_bytes_per_s', 'Host network receive', 'MiB/s', 1 / 2**20)]
    for key, title, unit, scale in resources:
        for aligned in (True, False):
            series = []
            for i, run in enumerate(runs):
                for node in sorted({point['node'] for point in run['resources']}):
                    points = [p for p in run['resources'] if p['node'] == node and isinstance(p.get(key), (int, float))
                              and ('aligned_s' in p) == aligned]
                    series.append((i, f'{run["name"]} / {node}',
                                   ([p.get('aligned_s' if aligned else 't_s') for p in points], [p[key] * scale for p in points]),
                                   [f'{escape(node)}; interval={p["interval_s"]:.4f}s; alignment error estimate (s)={p.get("alignment_error_s", "unknown")}' for p in points]))
            chart(title + ('' if aligned else ' (unaligned)'),
                  'Root training elapsed (s)' if aligned else 'Node-local runtime elapsed (s)', unit, series,
                  'Color identifies run; dash identifies role. Host metrics may repeat on colocated nodes; do not sum them. '
                  + ('Approximate clock alignment; resource readings span sampling intervals, not instantaneous events.' if aligned
                     else 'No clock mapping: these node-local times must not be interpreted as cross-device overlap.'), group_runs=True)
    for i, run in enumerate(runs):
        points = run['intervals']
        series = [(i, f'{run["name"]} / throughput',
                   ([p['elapsed'] for p in points], [p['throughput'] for p in points]), ['Root throughput'] * len(points))]
        for node in sorted({p['node'] for p in run['resources'] if 'aligned_s' in p}):
            points = [p for p in run['resources'] if p['node'] == node and 'aligned_s' in p]
            series.append((i, f'{run["name"]} / {node}',
                           ([p['aligned_s'] for p in points], [p['host_cpu_percent'] for p in points]),
                           [f'Host CPU; estimated alignment error={p["alignment_error_s"]:.6f}s' for p in points]))
        if len(series) > 1:
            chart(f'Throughput and device CPU — {run["name"]}', 'Root training elapsed (s)', 'Applied samples / s', series,
                  'Left axis: throughput; right axis: host CPU %. Shared approximate time basis, not proof of causation. '
                  'Sampling/display intervals limit temporal resolution; CPU includes other processes on that host.',
                  group_runs=True, secondary_y='Host CPU (%)')
    overhead_rows = []
    for run in runs:
        for node in run['config']['nodes']:
            node_id = node['id']
            components = {row['component']: row for row in run['overhead'] if row['node'] == node_id}
            role_records = [row for key, row in components.items()
                            if key.startswith('record:') and key not in ('record:resources', 'record:temperatures')]
            resource_records = [components[key] for key in ('resource_sampling', 'record:resources', 'record:temperatures')
                                if key in components]
            total_ms = lambda records: sum(row['wall_s'] for row in records if row['wall_s'] is not None) * 1000 if records else None
            role_ms, resource_ms = total_ms(role_records), total_ms(resource_records)
            role_calls = sum(row['calls'] for row in role_records)
            sampling = components.get('resource_sampling', {})
            writer_cpu = components.get('writer', {}).get('cpu_s')
            recording = run.get('recordings', {}).get(node_id, {})
            queue_wait = recording.get('queue_wait_s', components.get('queue_backpressure', {}).get('wall_s'))
            queue_events = recording.get('queue_full_events', components.get('queue_backpressure', {}).get('calls'))
            window = run.get('recording_windows', {}).get(node_id)
            fraction = role_ms / (window * 1000) * 100 if role_ms is not None and window and window > 0 else None
            overhead_rows.append(dict(run=run['name'], node=node_id, window_s=window, fraction=fraction,
                role_ms=role_ms, role_us=role_ms * 1000 / role_calls if role_calls else None,
                resource_ms=resource_ms,
                query_ms=sampling['wall_s'] * 1000 / sampling['calls'] if sampling.get('calls') else None,
                writer_ms=writer_cpu * 1000 if writer_cpu is not None else None,
                queue_ms=queue_wait * 1000 if queue_wait is not None else None,
                queue_events=queue_events))
    overhead_charts = []
    for key, title, axis in (
            ('role_ms', 'Role-thread recording', 'Total recording-call wall time (ms)'),
            ('resource_ms', 'Resource sampling', 'Total resource-sampler wall time (ms)'),
            ('writer_ms', 'CSV writer', 'Total writer-thread CPU time (ms)'),
            ('fraction', 'Recording-time fraction (not slowdown)', 'Role-thread recording / recorder lifetime (%)')):
        figure = go.Figure()
        labels = []
        for index, run in enumerate(runs):
            points = [row for row in overhead_rows if row['run'] == run['name']
                      and row[key] is not None and math.isfinite(row[key])]
            if not points:
                continue
            names = [escape(f'{row["run"]} / {row["node"]}') for row in points]
            labels.extend(names)
            figure.add_trace(go.Bar(
                x=[row[key] for row in points], y=names, orientation='h',
                name=escape(run['name']), marker_color=colors[index % len(colors)],
                texttemplate='%{x:.2f}' + ('%' if key == 'fraction' else ' ms'), textposition='auto',
                hovertemplate='%{y}<br>%{x:.3f}' + ('%' if key == 'fraction' else ' ms') + '<extra>%{fullData.name}</extra>'))
        if not labels:
            continue
        figure.update_layout(
            template='plotly_white', height=max(320, 28 * len(labels) + 120),
            margin=dict(t=30, b=65, l=260, r=250 if len(runs) > 1 else 35),
            xaxis=dict(title=axis, rangemode='tozero'),
            yaxis=dict(title='Run / node', categoryorder='array', categoryarray=labels,
                       autorange='reversed', automargin=True),
            showlegend=len(runs) > 1,
            legend=dict(x=1.02, xanchor='left', y=1, yanchor='top'))
        overhead_charts.append(dict(title=title, html=to_html(
            figure, full_html=False, include_plotlyjs=False,
            config=dict(displaylogo=False, responsive=True), div_id=f'overhead-{key}')))
    impact_rows = compare_instrumentation(runs, baseline_paths)
    impact_charts = []
    for key, title in (('training', 'Observed training slowdown'), ('launcher', 'Observed whole-run slowdown')):
        if not impact_rows:
            break
        figure = go.Figure(go.Bar(
            x=[row[key]['slowdown'] for row in impact_rows],
            y=[escape(row['label']) + f' (group {i + 1}, n={row["count"]})' for i, row in enumerate(impact_rows)],
            orientation='h', marker_color=['#64748b' if row['baseline'] else '#2563eb' for row in impact_rows],
            error_x=dict(type='data', symmetric=False,
                         array=[row[key]['high_pct'] - row[key]['slowdown'] for row in impact_rows],
                         arrayminus=[row[key]['slowdown'] - row[key]['low_pct'] for row in impact_rows]),
            texttemplate='%{x:.2f}%', textposition='auto',
            hovertemplate='%{y}<br>%{x:.3f}% vs baseline median<extra></extra>'))
        figure.update_layout(template='plotly_white', height=max(320, 45 * len(impact_rows) + 120),
                             margin=dict(t=30, b=65, l=360, r=35),
                             xaxis=dict(title='Duration change relative to baseline median (%)', rangemode='tozero'),
                             yaxis=dict(autorange='reversed', automargin=True))
        impact_charts.append(dict(title=title, html=to_html(figure, full_html=False, include_plotlyjs=False,
            config=dict(displaylogo=False, responsive=True), div_id=f'impact-{key}')))
    details = [dict(name=run['name'], path=run['path'], warnings=run['warnings'],
                    config=yaml.safe_dump(run['config'], sort_keys=False),
                    summary=json.dumps(run['summary'], indent=2)) for run in runs]
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).resolve().parent / 'templates'),
        autoescape=select_autoescape(['html']), trim_blocks=True, lstrip_blocks=True)
    return environment.get_template('run_report.html').render(
        runs=runs, rows=rows, setting_rows=setting_rows, charts=sections, details=details,
        overhead_rows=overhead_rows, overhead_charts=overhead_charts, impact_rows=impact_rows, impact_charts=impact_charts,
        plotly_js=get_plotlyjs() if sections or overhead_charts or impact_charts else None,
        has_resources=any(run['resources'] for run in runs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='saved run directories')
    parser.add_argument('--display-interval-s', type=float, default=0.1,
                        help='positive chart aggregation interval in seconds (default: 0.1)')
    parser.add_argument('--baseline', action='append', type=Path, default=[],
                        help='reference run, also listed as input; repeat for baseline repetitions')
    parser.add_argument('--output', required=True, type=Path, help='standalone HTML output')
    args = parser.parse_args(argv)
    try:
        if len({path.resolve() for path in args.runs}) != len(args.runs):
            raise ValueError('the same run directory was supplied more than once')
        runs = [read_run(path) for path in args.runs]
        if len({run['name'] for run in runs}) != len(runs):
            for run in runs:
                run['name'] = run['path']
        report = render_report(runs, args.display_interval_s, args.baseline)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    except ImportError as error:
        parser.exit(1, f'{error}. Install analysis dependencies: pip install -r requirements_analysis.txt\n')
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.exit(1, f'Cannot generate report: {error}\n')
    print(f'Wrote {args.output} ({len(runs)} runs); open directly in a browser.')
    for run in runs:
        for warning in run['warnings']:
            print(f'{run["name"]}: {warning}')


if __name__ == '__main__':
    main()
