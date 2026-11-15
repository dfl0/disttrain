"""Write node metrics to CSV in a background thread."""
import csv
from pathlib import Path
import queue
import threading
import time

SCHEMAS = {
    'updates': ('elapsed', 'version', 'samples', 'contributions', 'aggregate_samples',
                'aggregate_contributions', 'staleness_weighted_sum', 'staleness_max'),
    'evaluations': ('version', 'samples', 'root_elapsed', 'received', 'done', 'accuracy', 'loss', 'eval_samples'),
    'resources': ('t_s', 'interval_s', 'sample_end_s', 'host_cpu_percent', 'host_cpu_frequency_mhz',
                  'host_memory_percent', 'process_rss_bytes', 'host_throttled_now',
                  'host_throttled_since_boot', 'host_network_interface',
                  'host_network_tx_bytes_per_s', 'host_network_rx_bytes_per_s'),
    'temperatures': ('t_s', 'group', 'label', 'celsius'),
    'overhead': ('component', 'calls', 'wall_s', 'cpu_s'),
    'recording_window': ('elapsed_s',),
    'recording': ('records', 'queue_full_events', 'queue_wait_s', 'complete'),
}


class MetricsRecorder:
    def __init__(self, directory, *, queue_size=4096, flush_s=1.0, update_every=1, measure_overhead=True):
        if type(queue_size) is not int or queue_size <= 0 or flush_s <= 0:
            raise ValueError('metric queue size and flush interval must be positive')
        if type(update_every) is not int or update_every <= 0:
            raise ValueError('update_every must be a positive integer')
        self.update_every = update_every
        self.measure_overhead = measure_overhead
        self.last_update = None
        self.last_recorded_version = None
        self.overhead = {}
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue(maxsize=queue_size)
        self.flush_s = flush_s
        self.error = None
        self.closed = False
        self.queue_full_events = 0
        self.queue_wait_s = 0.0
        self.lock = threading.Lock()
        self.recording_started = time.perf_counter()
        self.recording_elapsed = None
        self.thread = threading.Thread(target=self._write, name='disttrain-csv', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f'metric recording failed: {self.error}') from self.error

    def update(self, *values):
        # keep the final update when it falls between recording intervals
        self.last_update = values
        version = values[1]
        if version % self.update_every == 0:
            self.record('updates', *values)
            self.last_recorded_version = version

    def measure(self, component, elapsed):
        if self.measure_overhead:
            with self.lock:
                calls, total = self.overhead.get(component, (0, 0.0))
                self.overhead[component] = (calls + 1, total + elapsed)

    def record(self, stream, *values):
        started = time.perf_counter() if self.measure_overhead else None
        try:
            if stream not in SCHEMAS or len(values) != len(SCHEMAS[stream]):
                raise ValueError(f'invalid {stream} metric row')
            if any(value is not None and type(value) not in (int, float, str, bool) for value in values):
                raise TypeError('metrics require immutable scalar values')
            if self.closed:
                raise RuntimeError('metric recorder is closed')
            self._enqueue((stream, values))
        finally:
            if started is not None:
                self.measure(f'record:{stream}', time.perf_counter() - started)

    def _enqueue(self, row, deadline=None):
        self.check()
        try:
            self.queue.put_nowait(row)
            return
        except queue.Full:
            started = time.perf_counter()
        try:
            while True:
                self.check()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('metric writer queue did not drain within shutdown budget')
                try:
                    self.queue.put(row, timeout=0.1)
                    return
                except queue.Full:
                    continue
        finally:
            with self.lock:
                self.queue_full_events += 1
                self.queue_wait_s += time.perf_counter() - started

    def _write(self):
        files, writers = {}, {}
        count = 0
        cpu_started = time.thread_time() if self.measure_overhead else None
        try:
            def write(stream, values):
                if stream not in writers:
                    file = (self.directory / f'{stream}.csv').open('x', newline='', encoding='utf-8', buffering=65536)
                    files[stream] = file
                    writers[stream] = csv.writer(file)
                    writers[stream].writerow(SCHEMAS[stream])
                writers[stream].writerow(values)

            next_flush = time.monotonic() + self.flush_s
            while True:
                try:
                    row = self.queue.get(timeout=max(0, next_flush - time.monotonic()))
                except queue.Empty:
                    row = 'flush'
                if row is None:
                    if self.measure_overhead:
                        writer_cpu = time.thread_time() - cpu_started
                        for component, (calls, elapsed) in sorted(self.overhead.items()):
                            write('overhead', (component, calls, elapsed, None))
                        write('overhead', ('writer', count, None, writer_cpu))
                        write('overhead', ('queue_backpressure', self.queue_full_events, self.queue_wait_s, None))
                    write('recording_window', (self.recording_elapsed,))
                    write('recording', (count, self.queue_full_events, self.queue_wait_s, True))
                    break
                if row != 'flush':
                    stream, values = row
                    write(stream, values)
                    count += 1
                if time.monotonic() >= next_flush:
                    for file in files.values():
                        file.flush()
                    next_flush = time.monotonic() + self.flush_s
        except Exception as error:
            self.error = error
        finally:
            for file in files.values():
                try:
                    file.close()
                except Exception as error:
                    self.error = error

    def close(self):
        if not self.closed:
            if self.last_update is not None and self.last_update[1] != self.last_recorded_version:
                self.record('updates', *self.last_update)
                self.last_recorded_version = self.last_update[1]
            self.recording_elapsed = time.perf_counter() - self.recording_started
            deadline = time.monotonic() + 10
            self._enqueue(None, deadline=deadline)
            self.closed = True
            self.thread.join(timeout=max(0, deadline - time.monotonic()))
            if self.thread.is_alive():
                raise TimeoutError('metric writer did not finish within 10 seconds')
        self.check()
