"""Worker and evaluator loops using ZeroMQ sockets and shared message framing."""
import time
import numpy as np
from data import load_datasets, limit_dataset, build_train_loader
from protocol import ParamSerializer, Msg, decode_packet, decode_assignment, encode_packet, encode_version
from protocol import encode_async_gradients, decode_async_params, decode_evaluation
from metrics import trace_span
from transport import send_message, receive_message, close_sockets


def traced_batches(loader, recorder):
    """Time loader startup and each next() call, not the consumer's computation."""
    if recorder is None or not recorder.trace_duration_ns:
        yield from loader
        return
    with trace_span(recorder, 'data_loader_start'):
        iterator = iter(loader)
    while True:
        with trace_span(recorder, 'load_batch'):
            try:
                batch = next(iterator)
            except StopIteration:
                return
        yield batch


def run_evaluator(pull_sock, push_sock, model, accuracy_threshold=0.90, time_limit_s=3600, batch_size=32, *, log, lifecycle=None, data_root="/mnt/data", download=False,
                  dataset_name="cifar10", sample_limit=None, num_workers=2, seed=0, recorder=None):
    import torch
    log("[EVALUATOR]")
    # -- load test data  --
    _, test_ds = load_datasets(root=data_root, download=download, name=dataset_name)

    test_ds = limit_dataset(test_ds, sample_limit, seed)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()


    serializer  = ParamSerializer(model)
    flat_params = np.empty(serializer.n_params, dtype=np.float32)

    test_acc     = 0
    model_version = -1
    stopping = False
    new_params   = False

    if lifecycle:
        lifecycle.ready()
        # Evaluation snapshots need not arrive every training round. Root enforces
        # optional round deadlines; evaluator keeps only its experiment time limit.
        lifecycle.deadline = None

    def request_stop():
        with trace_span(recorder, 'wait_stop'):
            if lifecycle:
                lifecycle.request_stop()
            send_message(push_sock, Msg.STOP)
            # Stay connected until root acknowledges STOP; otherwise a root PUSH can
            # block after the evaluator disappears. Root may have queued snapshots.
            while True:
                _, kind, payload = receive_message(pull_sock)
                if kind == bytes([Msg.STOP]):
                    return
                if kind != bytes([Msg.EVAL]):
                    raise ValueError('evaluator expected snapshot or STOP acknowledgement')
                decode_evaluation(payload, serializer.n_params)

    start_time = time.perf_counter()
    if recorder:
        recorder.clock_origin("evaluation", start_time)
    while True:
        if lifecycle:
            lifecycle.check()
        # -- drain all parameter pushes from root to get newest version --
        with trace_span(recorder, "receive_decode_snapshots"):
            drained = 0
            while pull_sock.poll(timeout=0):
                _, msg_type, payload = receive_message(pull_sock)
                if msg_type == bytes([Msg.STOP]):
                    stopping = True
                    break
                if msg_type != bytes([Msg.EVAL]):
                    raise ValueError('evaluator expected a snapshot or STOP')
                new_version, snapshot_samples, snapshot_elapsed, params = decode_evaluation(payload, serializer.n_params)
                drained += 1

                if new_version > model_version:
                    model_version = new_version
                    applied_samples, root_elapsed = snapshot_samples, snapshot_elapsed
                    flat_params[:] = params
                    new_params = True
                else:
                    log(f"WARNING: received non-monotonic eval version {new_version} (current {model_version}), skipping")

        if new_params:
            recv_time = time.perf_counter() - start_time
            log(f"drained {drained} pushed params")
            log(f"Newest version received: {model_version}. Starting evaluation...")

            with trace_span(recorder, "load_parameters", version=model_version):
                serializer.load_params(flat_params)
            new_params = False

            total_loss = 0.0
            total_samples = 0
            total_correct = 0

            with trace_span(recorder, "evaluation", version=model_version):
                model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in traced_batches(test_loader, recorder):
                        if lifecycle:
                            lifecycle.check()
                        with trace_span(recorder, 'evaluate_batch', version=model_version):
                            y_pred = model(X_batch)

                            predicted = torch.argmax(y_pred, 1)
                            total_correct += (predicted == y_batch).sum().item()

                            loss = loss_fn(y_pred, y_batch)
                            test_batch_size = X_batch.size(0)

                            total_loss += loss.item() * test_batch_size
                            total_samples += test_batch_size

                test_acc  = total_correct / total_samples
                test_loss = total_loss / total_samples

            done_time = time.perf_counter() - start_time
            if recorder:
                recorder.record('evaluations', model_version, applied_samples, root_elapsed,
                                 recv_time, done_time, test_acc * 100, test_loss, total_samples)
            log('*' * 100)
            log(f"Evaluation (recv'd after {recv_time:.2f}s, done after {done_time:.2f}s): version {model_version}, accuracy: {test_acc:.2%}, loss: {test_loss:.4f}")
            log('*' * 100)

            if lifecycle:
                lifecycle.evaluated(model_version)
            if stopping:
                break
            if test_acc >= accuracy_threshold or done_time >= time_limit_s:
                log("sending STOP message...")
                request_stop()
                break

        elif stopping:
            break
        elif time.perf_counter() - start_time >= time_limit_s:
            request_stop()
            break
        else:
            with trace_span(recorder, 'wait_snapshot'):
                if lifecycle:
                    # Readiness wakes immediately; expiry here is a successful time limit.
                    try:
                        lifecycle.wait([pull_sock], deadline=time.monotonic() + max(
                            0, time_limit_s - (time.perf_counter() - start_time)))
                    except TimeoutError:
                        request_stop()
                        break
                else:
                    pull_sock.poll(timeout=500)

    log("evaluator terminating...")

    close_sockets(pull_sock, push_sock)

    log("evaluator done")

def run_worker(dealer_sock, model, n_workers, batch_size=32, n_epochs=1, local_steps=1,
               *, worker_rank, seed=0, log, node_id, lifecycle=None, data_root="/mnt/data", download=False,
               dataset_name="cifar10", sample_limit=None, num_workers=2, mode="sync", trace=False, recorder=None):
    import torch
    send = send_message if lifecycle is None else lifecycle.send
    receive = receive_message if lifecycle is None else lifecycle.receive
    if type(n_epochs) is not int or n_epochs <= 0 or type(local_steps) is not int or local_steps <= 0:
        raise ValueError('epochs and local steps must be positive integers')
    torch.manual_seed(seed + worker_rank)
    train_ds, _ = load_datasets(root=data_root, download=download, name=dataset_name)
    train_ds = limit_dataset(train_ds, sample_limit, seed)
    loader = build_train_loader(train_ds, n_workers, worker_rank, batch_size, seed, num_workers)
    serializer = ParamSerializer(model)
    gradients = np.empty(serializer.n_params, dtype=np.float32)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    if mode not in ('sync', 'async'):
        raise ValueError('invalid worker mode')
    asynchronous = mode == 'async'
    expected_version = 0
    source_version = 0
    produced = dict(contributions=0, samples=0, partial_samples=0)
    if lifecycle:
        lifecycle.work_report = produced

    def receive_snapshot():
        nonlocal source_version
        with trace_span(recorder, 'wait_snapshot', request=expected_version):
            _, kind, payload = receive(dealer_sock)
        if kind == bytes([Msg.STOP]):
            return False
        if kind != bytes([Msg.ASYNC_PARAMS if asynchronous else Msg.PUSH_PARAMS]):
            raise ValueError('worker expected a root snapshot or STOP')
        with trace_span(recorder, 'decode_load_parameters', request=expected_version):
            if asynchronous:
                request, version, params = decode_async_params(payload, serializer.n_params)
                if request != expected_version or (request == 0 and version != 0) or (request > 0 and version <= source_version):
                    raise ValueError('worker received an unmatched async reply')
                source_version = version
                if trace:
                    log(f'[COMM] t={time.monotonic():.6f} event=reply request={request} version={version}')
            else:
                version, _, params = decode_packet(payload, serializer.n_params)
                if version != expected_version:
                    raise ValueError('worker received an unexpected model version')
            serializer.load_params(params)
        if lifecycle:
            if asynchronous:
                lifecycle.deadline = None
            else:
                lifecycle.begin_round()
        return True

    def send_gradients(samples):
        nonlocal expected_version
        request = expected_version + 1
        with trace_span(recorder, 'encode_gradients', request=request):
            serializer.flatten_grads_into(gradients)
            produced['contributions'] += 1
            produced['samples'] += samples
            produced['partial_samples'] = 0
            payload = (encode_async_gradients(request, [(worker_rank, request, source_version, samples)], gradients)
                       if asynchronous else encode_packet(expected_version, gradients, samples))
        if asynchronous and lifecycle:
            lifecycle.begin_round()
        with trace_span(recorder, 'send_gradients', request=request):
            send(dealer_sock, Msg.ASYNC_GRADS if asynchronous else Msg.PUSH_GRADS, payload)
        if asynchronous and trace:
            log(f'[COMM] t={time.monotonic():.6f} event=send request={request} source={source_version} samples={samples}')
        expected_version += 1

    try:
        if lifecycle:
            lifecycle.ready()
        with trace_span(recorder, 'register'):
            send(dealer_sock, Msg.REGISTER, node_id.encode())
            _, kind, payload = receive(dealer_sock)
            if kind != bytes([Msg.ASSIGN]) or decode_assignment(payload) != worker_rank:
                raise ValueError('assigned rank does not match configured global worker rank')
        log(f'[WORKER {worker_rank}] registered; {local_steps} microbatches per {"contribution" if asynchronous else "round"}')
        for epoch in range(n_epochs):
            log(f'Epoch {epoch + 1}/{n_epochs}')
            loader.sampler.set_epoch(epoch)
            model.train()
            steps = samples = 0
            for inputs, targets in traced_batches(loader, recorder):
                if lifecycle:
                    lifecycle.check()
                if steps == 0:
                    if not receive_snapshot():
                        return
                    with trace_span(recorder, 'zero_gradients'):
                        model.zero_grad(set_to_none=True)
                with trace_span(recorder, 'forward_backward',
                                version=source_version if asynchronous else expected_version,
                                request=expected_version + 1):
                    loss_fn(model(inputs), targets).backward()
                samples += inputs.size(0)
                produced['partial_samples'] = samples
                steps += 1
                if steps == local_steps:
                    send_gradients(samples)
                    steps = samples = 0
            if steps:
                send_gradients(samples)
        # Acknowledge the final committed round before declaring completion.
        if not receive_snapshot():
            return
        send(dealer_sock, Msg.ASYNC_DONE if asynchronous else Msg.DONE, encode_version(expected_version))
        with trace_span(recorder, 'wait_stop'):
            _, kind, _ = receive(dealer_sock)
        if kind != bytes([Msg.STOP]):
            raise ValueError('completed worker expected STOP')
    finally:
        close_sockets(dealer_sock)
