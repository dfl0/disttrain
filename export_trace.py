"""Export aligned operation spans, or one node with an independent time origin."""
import argparse
import csv
import json
from pathlib import Path
import sys

from metrics import SCHEMAS
from clock_alignment import ClockAlignment


def build_trace(directory, *, node=None):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    selected_node = node
    if selected_node is not None and selected_node not in summary['nodes']:
        raise ValueError(f'unknown node {selected_node!r}; available nodes: {", ".join(summary["nodes"])}')
    clocks = ClockAlignment(directory, summary['nodes'], local=summary['mode'] == 'local')
    aligned = selected_node is None and clocks.zero is not None
    if summary['mode'] != 'local' and selected_node is None and not aligned:
        raise ValueError('remote clocks have no calibration origin; select one node with --node NODE')
    events, spans, warnings = [], [], []
    calibration = []
    for pid, (node, details) in enumerate(summary['nodes'].items(), 1):
        if selected_node is not None and node != selected_node:
            continue
        path = directory / (details.get('metrics') or f'{node}.metrics') / 'spans.csv'
        if not path.exists():
            warnings.append(f'{node}: no captured spans')
            continue
        if aligned and not clocks.available(node):
            raise ValueError(f'{node}: missing clock calibration; use --node for an independent trace')
        if aligned:
            calibration.append(clocks.describe(node))
        if details.get('metrics_status') != 'complete':
            warnings.append(f'{node}: metrics collection incomplete; trace may be truncated')
        events.append(dict(ph='M', name='process_name', pid=pid, args=dict(name=f'{node} ({details["role"]})')))
        for tid, name in enumerate(('main', 'upstream')):
            events.append(dict(ph='M', name='thread_name', pid=pid, tid=tid, args=dict(name=name)))
        with path.open(newline='') as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != SCHEMAS['spans']:
                raise ValueError(f'{path}: unexpected span columns')
            for row in reader:
                start, end = int(row['start_ns']), int(row['end_ns'])
                if end < start or row['track'] not in ('main', 'upstream'):
                    raise ValueError(f'{path}: invalid span interval or track')
                args = dict(status=row['status'])
                for key in ('version', 'request', 'peer'):
                    if row[key]:
                        args[key] = row[key] if key == 'peer' else int(row[key])
                if aligned:
                    mapped, error, outside = clocks.project(node, start)
                    mapped_end, end_error, end_outside = clocks.project(node, end)
                    args.update(alignment_error_us=max(error, end_error) * 1e6,
                                outside_calibration=outside or end_outside)
                    if outside or end_outside:
                        warning = f'{node}: spans outside calibration interval use nearest offset; drift uncertainty unknown'
                        if warning not in warnings:
                            warnings.append(warning)
                    args['local_duration_us'] = (end - start) / 1000
                    args['root_elapsed_s'] = mapped
                    start, end = round(mapped * 1e9), round(mapped_end * 1e9)
                spans.append((start, end, dict(ph='X', name=row['operation'], cat='disttrain',
                                              pid=pid, tid=0 if row['track'] == 'main' else 1, args=args)))
    if not spans:
        raise ValueError('no spans found; enable metrics.trace_duration_s in a new run')
    # use one origin for all nodes and subtract before converting to microseconds
    origin = min(0, min(start for start, _, _ in spans)) if aligned else min(start for start, _, _ in spans)
    if aligned:
        events.append(dict(ph="i", s="g", name="Root training start", pid=0, tid=0, ts=-origin / 1000))
    if aligned and summary["mode"] != "local":
        warnings.append("Approximate root-relative alignment: endpoint uncertainty excludes unobserved nonlinear drift; raw local durations are retained in slice arguments.")
    for start, end, event in sorted(spans, key=lambda span: (span[0], -span[1])):
        events.append(dict(event, ts=(start - origin) / 1000, dur=(end - start) / 1000))
    return dict(traceEvents=events, displayTimeUnit='ms',
                otherData=dict(clock='root training elapsed (calibrated)' if aligned else 'node-local monotonic_ns' if selected_node is not None else 'same-host monotonic_ns',
                               calibration=calibration, trace_zero_root_elapsed_s=origin / 1e9 if aligned else None,
                               node=selected_node, origin_ns=origin, warnings=warnings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='launcher output directory')
    parser.add_argument('--node', help='export only this node with an independent time origin (no calibration required)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        trace = build_trace(args.run, node=args.node)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for warning in trace['otherData']['warnings']:
        print(f'Warning: {warning}', file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, separators=(',', ':')) + '\n')
    print(f'Trace: {args.output}')


if __name__ == '__main__':
    main()
