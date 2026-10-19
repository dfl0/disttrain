"""Evaluator behavior with actual tensors, DataLoader and in-process ZeroMQ."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
import zmq

from protocol import Msg, encode_packet
from training import run_evaluator


class EvaluatorTests(unittest.TestCase):
    def test_newest_snapshot_and_stop(self):
        context = zmq.Context()
        source, pull, push, stop = [context.socket(kind) for kind in (zmq.PUSH, zmq.PULL, zmq.PUSH, zmq.PULL)]
        try:
            pull.bind('inproc://eval')
            source.connect('inproc://eval')
            stop.bind('inproc://stop')
            stop.setsockopt(zmq.RCVTIMEO, 2000)
            push.connect('inproc://stop')
            for version in (1, 3, 2):
                source.send_multipart([bytes([Msg.EVAL]), encode_packet(version, np.array([version, 0], dtype=np.float32))])
            source.send_multipart([bytes([Msg.STOP]), b''])
            self.assertTrue(pull.poll(2000))
            dataset = torch.utils.data.TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
            model = torch.nn.Linear(1, 2, bias=False)
            # Only the source of data is replaced; computation and I/O are real.
            with patch('training.load_datasets', return_value=(dataset, dataset)):
                run_evaluator(pull, push, model, accuracy_threshold=0, time_limit_s=0,
                              log=print)
            np.testing.assert_array_equal(model.weight.detach().numpy(), [[3], [0]])
            self.assertEqual(stop.poll(0), 0)  # root already initiated completion
            self.assertTrue(pull.closed and push.closed)
        finally:
            for sock in (source, pull, push, stop):
                sock.close(0)
            context.term()
