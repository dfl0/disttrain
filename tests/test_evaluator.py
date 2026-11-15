"""Evaluator behavior with actual tensors, DataLoader and in-process ZeroMQ."""
import unittest
import tempfile
import csv
from pathlib import Path
from metrics import MetricsRecorder
from unittest.mock import patch

import numpy as np
import torch
import zmq

from protocol import Msg, encode_evaluation
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
                source.send_multipart([bytes([Msg.EVAL]), encode_evaluation(version, version * 32, version * 0.123456789, np.array([version, 0], dtype=np.float32))])
            source.send_multipart([bytes([Msg.STOP]), b''])
            self.assertTrue(pull.poll(2000))
            dataset = torch.utils.data.TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
            model = torch.nn.Linear(1, 2, bias=False)
            # use synthetic data with real tensors and sockets
            with tempfile.TemporaryDirectory() as directory, patch('training.load_datasets', return_value=(dataset, dataset)):
                recorder = MetricsRecorder(Path(directory) / 'metrics')
                run_evaluator(pull, push, model, accuracy_threshold=0, time_limit_s=0,
                              log=print, recorder=recorder)
                recorder.close()
                with (recorder.directory / 'evaluations.csv').open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 1)
                self.assertEqual((rows[0]['version'], rows[0]['samples']), ('3', '96'))
                self.assertAlmostEqual(float(rows[0]['root_elapsed']), 3 * 0.123456789)
            np.testing.assert_array_equal(model.weight.detach().numpy(), [[3], [0]])
            self.assertEqual(stop.poll(0), 0)  # root already initiated completion
            self.assertTrue(pull.closed and push.closed)
        finally:
            for sock in (source, pull, push, stop):
                sock.close(0)
            context.term()
