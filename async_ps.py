"""Handle async and mixed-policy aggregation."""
import time
import numpy as np
from ps import ParameterServer
from metrics import trace_span
from protocol import (Msg, MAX_VERSION, encode_async_gradients, decode_async_gradients,
                      encode_async_params, decode_async_params, decode_version, encode_version)


class AsyncParameterServer(ParameterServer):
    synchronous = False
    gradient_message = Msg.ASYNC_GRADS
    snapshot_message = Msg.ASYNC_PARAMS
    done_message = Msg.ASYNC_DONE

    def __init__(self, model, children_ids, sync, *args, worker_groups, reference_samples, trace=False, **kwargs):
        self.synchronous = sync
        super().__init__(model, children_ids, sync, *args, **kwargs)
        if type(reference_samples) is not int or reference_samples <= 0:
            raise ValueError('reference sample count must be positive')
        self.worker_groups = {child: set(ranks) for child, ranks in worker_groups.items()}
        if set(self.worker_groups) != set(self.children_ids) or any(not ranks for ranks in self.worker_groups.values()):
            raise ValueError('worker groups must cover every child')
        ranks = [rank for group in self.worker_groups.values() for rank in group]
        if len(set(ranks)) != len(ranks):
            raise ValueError('worker groups must be disjoint')
        self.reference_samples = reference_samples
        self.trace_enabled = trace
        self.requests = dict.fromkeys(self.children_ids, 0)
        self.contributions = dict.fromkeys(ranks, 0)
        self.sources = dict.fromkeys(ranks, 0)
        self.pending = []
        self.inflight = []
        self.upstream_request = 0
        self.applied_contributions = 0
        self.weighted_staleness = 0
        self.max_staleness = 0

    def _trace(self, event, **fields):
        if self.trace_enabled:
            self.log(f'[COMM] t={time.monotonic():.6f} event={event} {fields}')

    def _decode_initial_snapshot(self, payload):
        request, version, params = decode_async_params(payload, self.flat_params.size)
        if request != 0:
            raise ValueError('initial snapshot request must be zero')
        return version, params

    def _broadcast_snapshot(self):
        # broadcast only the initial snapshot
        with trace_span(self.recorder, 'encode_parameters', version=self.version, request=0):
            payload = encode_async_params(0, self.version, self.flat_params)
        for child in self.children_ids:
            with trace_span(self.recorder, 'send_parameters', peer=child, version=self.version, request=0):
                self.send(self._router, self.snapshot_message, payload, identity=child.encode())

    def _event_loop(self):
        if self.lifecycle and not self.is_root:
            self.lifecycle.deadline = None
        super()._event_loop()

    def _accept_contribution(self, child, payload):
        if self.stopped or child not in self.registered or child in self.finished or child in self.received:
            raise ValueError('unknown, completed or busy async child')
        with trace_span(self.recorder, 'decode_gradients', peer=child):
            request, records, gradients = decode_async_gradients(payload, self.flat_params.size)
            if request != self.requests[child] + 1:
                raise ValueError('duplicate or out-of-order async request')
            if self.child_ranks[child] >= 0 and (len(records) != 1 or records[0][1] != request):
                raise ValueError('worker must send exactly its own sequential contribution')
            for rank, contribution, source, samples in records:
                if (rank not in self.worker_groups[child] or contribution != self.contributions[rank] + 1
                        or source != self.sources[rank] or source > self.version):
                    raise ValueError('invalid async membership, contribution ID or source version')
        count = sum(r[3] for r in records)
        if self.sample_count + count > MAX_VERSION:
            raise ValueError('pending sample count overflow')
        with trace_span(self.recorder, 'aggregate_gradients', peer=child, version=self.version):
            with np.errstate(over='ignore', invalid='ignore'):
                combined = self.gradient_sum + gradients
            if not np.isfinite(combined).all():
                raise ValueError('pending gradient sum overflow')
        # keep the child busy until its contribution is applied
        self.requests[child] = request
        for rank, contribution, source, samples in records:
            self.contributions[rank] = contribution
        self.gradient_sum[:] = combined
        self.sample_count += count
        self.pending.extend(records)
        self.received.add(child)
        self._trace('receive', child=child, request=request, records=records)
        self._advance_pending()

    def _advance_pending(self):
        if not self.pending or self.inflight:
            return
        if self.synchronous and len(self.received) + len(self.finished) != len(self.children_ids):
            return
        if self.is_root:
            self._apply_pending()
        else:
            self._forward_pending()

    def _apply_pending(self):
        if self.version == MAX_VERSION:
            raise ValueError('root version overflow')
        with trace_span(self.recorder, 'compute_update', version=self.version + 1), np.errstate(over='ignore', invalid='ignore'):
            updated = self.flat_params - self.lr * (self.gradient_sum / self.reference_samples)
        if not np.isfinite(updated).all():
            raise ValueError('non-finite root update')
        for rank, contribution, source, samples in self.pending:
            age = self.version - source
            self.weighted_staleness += age * samples
            self.max_staleness = max(self.max_staleness, age)
            self.sources[rank] = self.version + 1
        self.flat_params[:] = updated
        self.version += 1
        self.total_samples += self.sample_count
        self.applied_contributions += len(self.pending)
        self.update_elapsed = time.perf_counter() - self.training_started
        if self.recorder:
            self.recorder.update(self.update_elapsed, self.version, self.total_samples,
                                 self.applied_contributions, self.sample_count, len(self.pending),
                                 self.weighted_staleness, self.max_staleness)
        if self.lifecycle:
            self.lifecycle.begin_round()
        self._trace('apply', version=self.version, records=self.pending)
        if self.version == 1 or self.version % self.log_interval == 0:
            self._log_progress()
        children = [child for child in self.children_ids if child in self.received]
        self.pending = []
        self._reset_round()
        if self.max_rounds is not None and self.version >= self.max_rounds:
            self._finish_training()
            return
        for child in children:
            self._reply(child, self.requests[child])
        if self.version % self.eval_interval == 0:
            self._push_eval()

    def _reply(self, child, request):
        with trace_span(self.recorder, 'encode_parameters', version=self.version, request=request):
            payload = encode_async_params(request, self.version, self.flat_params)
        with trace_span(self.recorder, 'send_parameters', peer=child, version=self.version, request=request):
            self.send(self._router, self.snapshot_message, payload, identity=child.encode())
        self._trace('reply', child=child, request=request, version=self.version)

    def _forward_pending(self):
        if not self.pending or self.inflight:
            return
        request = self.upstream_request + 1
        self._start_upstream_span()
        with trace_span(self.recorder, 'encode_gradients', request=request):
            payload = encode_async_gradients(request, self.pending, self.gradient_sum)
        if self.lifecycle:
            self.lifecycle.begin_round()  # child traffic must not reset this deadline
        with trace_span(self.recorder, 'send_gradients', peer=self.parent, request=request):
            self.send(self._dealer, self.gradient_message, payload)
        self.upstream_request = request
        self.inflight, self.pending = self.pending, []
        self.gradient_sum.fill(0)
        self.sample_count = 0
        self._trace('forward', request=request, records=self.inflight)

    def _dispatch_parent_push(self, payload):
        with trace_span(self.recorder, 'decode_load_parameters', peer=self.parent):
            request, version, params = decode_async_params(payload, self.flat_params.size)
            if not self.inflight or request != self.upstream_request or version <= self.version:
                raise ValueError('unmatched or non-monotonic async reply')
            self.flat_params[:] = params
            self.version = version
        self._end_upstream_span()
        if self.lifecycle:
            self.lifecycle.deadline = None
        by_rank = {rank: child for child, rank in self.child_ranks.items()}
        for rank, contribution, source, samples in self.inflight:
            child = by_rank[rank]
            self.sources[rank] = version
            self._reply(child, contribution)
            self.received.remove(child)
        self.inflight = []
        self._advance_pending()

    def _accept_done(self, child, payload):
        if (self.stopped or child not in self.registered or child in self.finished
                or child in self.received or decode_version(payload) != self.requests[child]):
            raise ValueError('invalid async completion')
        self.finished.add(child)
        if len(self.finished) == len(self.children_ids):
            if self.is_root:
                self._finish_training()
            elif not self.pending and not self.inflight:
                self.send(self._dealer, self.done_message, encode_version(self.upstream_request))
                if self.lifecycle:
                    self.lifecycle.deadline = None
        else:
            # DONE can release a pending barrier
            self._advance_pending()

    def _log_progress(self):
        mean_age = self.weighted_staleness / self.total_samples if self.total_samples else 0
        mean_group = self.applied_contributions / self.version if self.version else 0
        self.log(f'[TRAIN] update={self.version} applied_samples={self.total_samples} '
                 f'contributions={self.applied_contributions} mean_aggregate={mean_group:.2f} '
                 f'staleness_mean={mean_age:.2f} staleness_max={self.max_staleness} '
                 f'elapsed={time.perf_counter() - self.training_started:.2f}s')
        self.last_logged_version = self.version
