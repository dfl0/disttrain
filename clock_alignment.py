"""Align node timestamps using startup and shutdown clock samples.

The error estimate does not account for nonlinear clock drift.
"""
import csv
from pathlib import Path

from metrics import SCHEMAS


def clock_rows(path, stream):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != SCHEMAS[stream]:
            raise ValueError(f'{path}: unexpected clock columns')
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}: incomplete clock record')
            rows.append({key: value if key in ('name', 'node', 'phase') else int(value)
                         for key, value in row.items()})
        return rows


class ClockAlignment:
    def __init__(self, directory, nodes, *, local=False):
        self.local = local
        self.root = next((node for node, spec in nodes.items() if spec['role'] == 'root_ps'), None)
        self.origins, self.anchors = {}, {}
        directory = Path(directory)
        for node, spec in nodes.items():
            folder = directory / (spec.get('metrics') or f'{node}.metrics')
            self.origins[node] = {row['name']: row for row in clock_rows(folder / 'clock_origins.csv', 'clock_origins')}
            if node == self.root:
                for row in clock_rows(folder / 'clock_calibration.csv', 'clock_calibration'):
                    if row['phase'] not in ('startup', 'shutdown') or row['uncertainty_ns'] < 0:
                        raise ValueError('invalid clock calibration')
                    phases = self.anchors.setdefault(row['node'], {})
                    if row['phase'] in phases:
                        raise ValueError('duplicate clock calibration phase')
                    phases[row['phase']] = row
        self.zero = self.origins.get(self.root, {}).get('training', {}).get('monotonic_ns')
        for phases in self.anchors.values():
            if 'startup' in phases and 'shutdown' in phases:
                first, last = phases['startup'], phases['shutdown']
                if last['local_ns'] <= first['local_ns'] or last['root_ns'] <= first['root_ns']:
                    raise ValueError('non-increasing clock calibration anchors')

    def available(self, node):
        return self.zero is not None and (self.local or node == self.root or bool(self.anchors.get(node)))

    def project(self, node, timestamp_ns):
        """Return root-training seconds, estimated error seconds, and outside-window flag."""
        if not self.available(node):
            return None
        base_error = self.origins[self.root]['training']['uncertainty_ns']
        if node == self.root or self.local:
            return (timestamp_ns - self.zero) / 1e9, base_error / 1e9, False
        anchors = sorted(self.anchors[node].values(), key=lambda row: row['local_ns'])
        first, last = anchors[0], anchors[-1]
        outside = timestamp_ns < first['local_ns'] or timestamp_ns > last['local_ns']
        if len(anchors) == 1 or outside:
            # use the nearest offset outside the measured interval
            anchor = min(anchors, key=lambda row: abs(timestamp_ns - row['local_ns']))
            offset = anchor['root_ns'] - anchor['local_ns']
            error = anchor['uncertainty_ns']
        else:
            weight = (timestamp_ns - first['local_ns']) / (last['local_ns'] - first['local_ns'])
            offset = ((first['root_ns'] - first['local_ns']) * (1 - weight)
                      + (last['root_ns'] - last['local_ns']) * weight)
            error = first['uncertainty_ns'] * (1 - weight) + last['uncertainty_ns'] * weight
        # subtract before converting to float to preserve precision
        return ((timestamp_ns - self.zero) + offset) / 1e9, (error + base_error) / 1e9, outside

    def elapsed(self, node, origin, seconds):
        anchor = self.origins.get(node, {}).get(origin)
        if anchor is None:
            return None
        result = self.project(node, anchor['monotonic_ns'] + round(seconds * 1e9))
        if result is None:
            return None
        value, error, outside = result
        return value, error + anchor['uncertainty_ns'] / 1e9, outside

    def describe(self, node):
        phases = self.anchors.get(node, {})
        first, last = phases.get('startup'), phases.get('shutdown')
        method = ('unavailable' if not self.available(node) else 'same host' if self.local or node == self.root
                  else 'linear estimate' if first and last else 'offset only; drift unknown')
        drift = None
        resolved = None
        if first and last:
            change = (last['root_ns'] - last['local_ns']) - (first['root_ns'] - first['local_ns'])
            drift = change / (last['local_ns'] - first['local_ns']) * 1e6
            resolved = abs(change) > first['uncertainty_ns'] + last['uncertainty_ns']
        return dict(node=node, method=method,
                    start_error_ms=first['uncertainty_ns'] / 1e6 if first else None,
                    end_error_ms=last['uncertainty_ns'] / 1e6 if last else None,
                    correction_ppm=drift, drift_resolved=resolved)
