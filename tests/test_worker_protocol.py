"""Test worker gradients and messages using local sockets."""
import multiprocessing
import unittest

import numpy as np
import torch
import zmq

from protocol import encode_async_params, decode_async_gradients
from protocol import Msg, encode_assignment, decode_packet, encode_packet, decode_version


def worker_process(endpoint, samples, local_steps, batch_size, asynchronous=False):
    import training
    dataset = torch.utils.data.TensorDataset(torch.ones(samples, 1), torch.zeros(samples, dtype=torch.long))
    # use synthetic data with the real loader
    training.load_datasets = lambda **kwargs: (dataset, dataset)
    context = zmq.Context()
    dealer = context.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, b'worker')
    dealer.connect(endpoint)
    try:
        training.run_worker(dealer, torch.nn.Linear(1, 2, bias=False), n_workers=1,
                            batch_size=batch_size, local_steps=local_steps, worker_rank=0,
                            log=print, node_id='worker', mode='async' if asynchronous else 'sync')
    finally:
        dealer.close(0)
        context.term()


class WorkerProtocolTests(unittest.TestCase):
    def test_async_partial_windows_and_version_jumps(self):
        self.check_windows(local_steps=2, samples=5, expected=[4, 1], batch_size=2, asynchronous=True)

    def check_windows(self, local_steps, samples, expected, batch_size=1, asynchronous=False):
        context = zmq.Context()
        router = context.socket(zmq.ROUTER)
        router.setsockopt(zmq.RCVTIMEO, 20000)
        port = router.bind_to_random_port('tcp://127.0.0.1')
        process = multiprocessing.get_context('spawn').Process(
            target=worker_process, args=(f'tcp://127.0.0.1:{port}', samples, local_steps, batch_size, asynchronous))
        process.start()
        try:
            self.assertEqual(router.recv_multipart(), [b'worker', bytes([Msg.REGISTER]), b'worker'])
            router.send_multipart([b'worker', bytes([Msg.ASSIGN]), encode_assignment(0)])
            for version, count in enumerate(expected):
                # send one snapshot per window
                router.send_multipart([b'worker', bytes([Msg.ASYNC_PARAMS if asynchronous else Msg.PUSH_PARAMS]),
                                       encode_async_params(version, version * 3, np.zeros(2, dtype=np.float32)) if asynchronous
                                       else encode_packet(version, np.zeros(2, dtype=np.float32))])
                identity, kind, payload = router.recv_multipart()
                self.assertEqual((identity, kind), (b'worker', bytes([Msg.ASYNC_GRADS if asynchronous else Msg.PUSH_GRADS])))
                if asynchronous:
                    request, records, gradient = decode_async_gradients(payload, 2)
                    self.assertEqual(request, version + 1)
                    self.assertEqual(records, [(0, version + 1, version * 3, count)])
                    actual_version, count_received = version, records[0][3]
                else:
                    actual_version, count_received, gradient = decode_packet(payload, 2, gradients=True)
                self.assertEqual(count_received, count)
                self.assertEqual(actual_version, version)
                # expected gradient is [-0.5, 0.5] per sample
                np.testing.assert_allclose(gradient, np.array([-0.5, 0.5]) * count)
            if expected:
                router.send_multipart([b'worker', bytes([Msg.ASYNC_PARAMS if asynchronous else Msg.PUSH_PARAMS]),
                                       encode_async_params(len(expected), len(expected) * 3, np.zeros(2, dtype=np.float32)) if asynchronous
                                       else encode_packet(len(expected), np.zeros(2, dtype=np.float32))])
                identity, kind, payload = router.recv_multipart()
                self.assertEqual((identity, kind), (b'worker', bytes([Msg.ASYNC_DONE if asynchronous else Msg.DONE])))
                self.assertEqual(decode_version(payload), len(expected))
            router.send_multipart([b'worker', bytes([Msg.STOP]), b''])
            process.join(20)
            self.assertEqual(process.exitcode, 0, 'worker did not terminate after STOP')
            self.assertEqual(router.poll(0), 0, 'unexpected extra gradient')
        finally:
            if process.is_alive():
                process.terminate()
            process.join(5)
            process.close()
            router.close(0)
            context.term()

    def test_gradient_windows_and_partial_epoch(self):
        for steps, batches, expected in [(1, 2, [1, 1]), (2, 4, [2, 2]), (2, 3, [2, 1])]:
            with self.subTest(local_steps=steps, batches=batches):
                self.check_windows(steps, batches, expected)

    def test_stop_before_initial_model(self):
        self.check_windows(1, 1, [])

    def test_partial_batch_is_sample_weighted(self):
        self.check_windows(2, 5, [4, 1], batch_size=2)

    def test_sampler_coverage_padding_and_epoch_shuffle(self):
        from data import build_train_loader
        for size in (12, 10):
            dataset = torch.utils.data.TensorDataset(torch.arange(size))
            loaders = [build_train_loader(dataset, 3, rank, 3, seed=7) for rank in range(3)]
            shards = [list(loader.sampler) for loader in loaders]
            flattened = [index for shard in shards for index in shard]
            self.assertEqual(set(flattened), set(range(size)))
            self.assertEqual([len(shard) for shard in shards], [4, 4, 4])
            self.assertEqual(len(flattened) - len(set(flattened)), 12 - size)
            self.assertTrue(all(len(loader) == 2 for loader in loaders))  # retain batch tail
            self.assertEqual(shards, [list(loader.sampler) for loader in loaders])
            for loader in loaders:
                loader.sampler.set_epoch(1)
            self.assertNotEqual(shards, [list(loader.sampler) for loader in loaders])

    def test_subset_and_zero_loader_processes(self):
        from data import limit_dataset, build_train_loader
        dataset = torch.utils.data.TensorDataset(torch.arange(20))
        before = torch.get_rng_state().clone()
        subset = limit_dataset(dataset, 7, seed=3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(subset), 7)
        self.assertEqual(subset.indices, limit_dataset(dataset, 7, seed=3).indices)
        self.assertEqual(len(set(subset.indices)), 7)
        self.assertIs(limit_dataset(dataset, 100, seed=3), dataset)
        loader = build_train_loader(subset, 1, 0, 3, num_workers=0)
        self.assertEqual(loader.num_workers, 0)
        self.assertEqual(sum(len(batch[0]) for batch in loader), 7)
