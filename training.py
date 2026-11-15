"""Worker and evaluator loops using ZeroMQ sockets and shared message framing."""
import time
import numpy as np
from data import load_datasets, limit_dataset, build_train_loader
from protocol import ParamSerializer, Msg, decode_packet, decode_assignment, encode_packet, encode_version, decode_evaluation
from timetree import TimeTree
from transport import send_message, receive_message, close_sockets


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

    timer = TimeTree(print_func=log)

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
    while True:
        if lifecycle:
            lifecycle.check()
        # -- drain all parameter pushes from root to get newest version --
        with timer.measure("drain"):
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

            with timer.measure("deserialize_params"):
                serializer.load_params(flat_params)
            new_params = False

            total_loss = 0.0
            total_samples = 0
            total_correct = 0

            with timer.measure("eval"):
                model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        if lifecycle:
                            lifecycle.check()
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
               dataset_name="cifar10", sample_limit=None, num_workers=2):
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
    expected_version = 0
    timer = TimeTree(print_func=log)

    def receive_snapshot():
        with timer.measure('wait_recv'):
            _, kind, payload = receive(dealer_sock)
        if kind == bytes([Msg.STOP]):
            return False
        if kind != bytes([Msg.PUSH_PARAMS]):
            raise ValueError('worker expected a root snapshot or STOP')
        version, _, params = decode_packet(payload, serializer.n_params)
        if version != expected_version:
            raise ValueError('worker received an unexpected model version')
        serializer.load_params(params)
        if lifecycle:
            lifecycle.begin_round()
        return True

    try:
        if lifecycle:
            lifecycle.ready()
        send(dealer_sock, Msg.REGISTER, node_id.encode())
        _, kind, payload = receive(dealer_sock)
        if kind != bytes([Msg.ASSIGN]) or decode_assignment(payload) != worker_rank:
            raise ValueError('assigned rank does not match configured global worker rank')
        log(f'[WORKER {worker_rank}] registered; {local_steps} microbatches per round')
        for epoch in range(n_epochs):
            log(f'Epoch {epoch + 1}/{n_epochs}')
            loader.sampler.set_epoch(epoch)
            model.train()
            steps = samples = 0
            for inputs, targets in loader:
                if lifecycle:
                    lifecycle.check()
                if steps == 0:
                    if not receive_snapshot():
                        return
                    model.zero_grad(set_to_none=True)
                with timer.measure('step'):
                    loss_fn(model(inputs), targets).backward()
                samples += inputs.size(0)
                steps += 1
                if steps == local_steps:
                    serializer.flatten_grads_into(gradients)
                    send(dealer_sock, Msg.PUSH_GRADS,
                                 encode_packet(expected_version, gradients, samples))
                    expected_version += 1
                    steps = samples = 0
            if steps:
                serializer.flatten_grads_into(gradients)
                send(dealer_sock, Msg.PUSH_GRADS,
                             encode_packet(expected_version, gradients, samples))
                expected_version += 1
        # Acknowledge the final committed round before declaring completion.
        if not receive_snapshot():
            return
        send(dealer_sock, Msg.DONE, encode_version(expected_version))
        _, kind, _ = receive(dealer_sock)
        if kind != bytes([Msg.STOP]):
            raise ValueError('completed worker expected STOP')
    finally:
        close_sockets(dealer_sock)
        timer.print_tree()
