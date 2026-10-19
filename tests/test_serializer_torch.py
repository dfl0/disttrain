"""Real tensor checks: python3 -m unittest discover -s tests -p test_serializer_torch.py -v"""
import unittest
import warnings
import numpy as np
import torch
from protocol import ParamSerializer


class RealTensorSerializerTests(unittest.TestCase):
    def test_roundtrip_backward_missing_gradient_and_readonly_source(self):
        model = torch.nn.Linear(2, 1)
        serializer = ParamSerializer(model)
        model(torch.ones(1, 2)).sum().backward()
        gradients = np.empty(serializer.n_params, dtype=np.float32)
        serializer.flatten_grads_into(gradients)
        np.testing.assert_array_equal(gradients, [1, 1, 1])
        model.bias.grad = None
        serializer.flatten_grads_into(gradients)
        np.testing.assert_array_equal(gradients, [1, 1, 0])
        source = np.frombuffer(np.array([2, 3, 4], dtype=np.float32).tobytes(), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            serializer.load_params(source)
        output = np.empty(3, dtype=np.float32)
        serializer.flatten_params_into(output)
        np.testing.assert_array_equal(output, source)
        np.testing.assert_array_equal(source, [2, 3, 4])

    def test_tensor_dtype_layout_device_and_noncontiguous_gradient_rejections(self):
        tensors = [torch.ones(2, 2, dtype=torch.float64), torch.ones(2, 2).t(),
                   torch.ones(2, 2, device='meta'), torch.eye(2).to_sparse()]
        for tensor in tensors:
            model = torch.nn.Module(); model.register_parameter('weight', torch.nn.Parameter(tensor))
            with self.subTest(tensor=tensor), self.assertRaisesRegex(ValueError, 'weight.*expected.*actual'):
                ParamSerializer(model)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False),
                                    torch.nn.Linear(2, 2, bias=False))
        model[0].weight.grad = torch.ones(2, 2)
        gradient = torch.arange(4, dtype=torch.float32).reshape(2, 2).t()
        model[1].weight.grad = gradient
        # Construction and parameter transfers do not use existing gradients.
        serializer = ParamSerializer(model)
        source = np.arange(8, dtype=np.float32)
        serializer.load_params(source)
        output = np.empty(8, dtype=np.float32)
        serializer.flatten_params_into(output)
        np.testing.assert_array_equal(output, source)
        self.assertIs(model[1].weight.grad, gradient)
        torch.testing.assert_close(gradient, torch.tensor([[0., 2.], [1., 3.]]))
        # A bad later gradient must reject before writing even the first slot.
        output.fill(42)
        with self.assertRaisesRegex(ValueError, 'gradient 1.weight'):
            serializer.flatten_grads_into(output)
        np.testing.assert_array_equal(output, [42] * 8)

    def test_parameter_changes_rejected_before_writes(self):
        for change in ('replace', 'reshape', 'add', 'remove', 'reorder', 'dtype'):
            model = torch.nn.Linear(2, 1)
            serializer = ParamSerializer(model)
            if change == 'replace':
                model.bias = torch.nn.Parameter(torch.ones(1))
            elif change == 'reshape':
                model.weight.data = model.weight.data.reshape(2, 1)
            elif change == 'add':
                model.register_parameter('extra', torch.nn.Parameter(torch.ones(1)))
            elif change == 'remove':
                del model.bias
            elif change == 'reorder':
                weight = model.weight
                del model.weight
                model.register_parameter('weight', weight)
            else:
                model.bias.data = model.bias.data.double()
            source = np.full(3, 42, dtype=np.float32)
            original = [p.detach().clone() for p in model.parameters()]
            for operation in (serializer.flatten_params_into, serializer.flatten_grads_into, serializer.load_params):
                with self.subTest(change=change, operation=operation.__name__):
                    with self.assertRaises(ValueError):
                        operation(source)
                    for parameter, before in zip(model.parameters(), original):
                        torch.testing.assert_close(parameter, before)
                    np.testing.assert_array_equal(source, [42] * 3)

    def test_flat_array_contract(self):
        model = torch.nn.Linear(2, 1)
        serializer = ParamSerializer(model)
        original = [p.detach().clone() for p in model.parameters()]
        invalid = [np.ones(2, dtype=np.float32), np.ones((1, 3), dtype=np.float32),
                   np.ones(3, dtype=np.float64), np.ones(6, dtype=np.float32)[::2],
                   np.ones(3, dtype=np.dtype(np.float32).newbyteorder('S'))]
        for operation in (serializer.flatten_params_into, serializer.flatten_grads_into, serializer.load_params):
            for array in invalid:
                before = array.copy()
                with self.subTest(operation=operation.__name__, shape=array.shape, dtype=array.dtype):
                    with self.assertRaisesRegex(ValueError, 'flat array'):
                        operation(array)
                    np.testing.assert_array_equal(array, before)
                    for parameter, saved in zip(model.parameters(), original):
                        torch.testing.assert_close(parameter, saved)
        readonly = np.ones(3, dtype=np.float32)
        readonly.flags.writeable = False
        for operation in (serializer.flatten_params_into, serializer.flatten_grads_into):
            with self.assertRaisesRegex(ValueError, 'flat array'):
                operation(readonly)
        np.testing.assert_array_equal(readonly, [1, 1, 1])

    def test_registered_buffers_and_batchnorm_policy(self):
        from models import disable_state
        for persistent in (False, True):
            model = torch.nn.Linear(2, 1)
            model.register_buffer('extra', torch.ones(1), persistent=persistent)
            with self.assertRaisesRegex(ValueError, 'extra'):
                ParamSerializer(model)
        model = torch.nn.Linear(2, 1)
        serializer = ParamSerializer(model)
        original = model.weight.detach().clone()
        model.register_buffer('late_buffer', torch.ones(1))
        with self.assertRaisesRegex(ValueError, 'late_buffer'):
            serializer.load_params(np.zeros(3, dtype=np.float32))
        torch.testing.assert_close(model.weight, original)
        bn = torch.nn.BatchNorm2d(2)
        with self.assertRaises(ValueError): ParamSerializer(bn)
        bn.apply(disable_state)
        serializer = ParamSerializer(bn)
        serializer.flatten_params_into(np.empty(serializer.n_params, dtype=np.float32))
        bn.track_running_stats = True
        with self.assertRaises(ValueError):
            serializer.flatten_params_into(np.empty(serializer.n_params, dtype=np.float32))

    def test_repository_resnet_accepts_disabled_state_roundtrip(self):
        from models import ResNet20CIFAR, BasicBlock, disable_state
        model = ResNet20CIFAR(BasicBlock, [3, 3, 3])
        model.apply(disable_state)
        serializer = ParamSerializer(model)
        output = np.empty(serializer.n_params, dtype=np.float32)
        serializer.flatten_params_into(output)
        serializer.load_params(output)
        result = np.empty_like(output)
        serializer.flatten_params_into(result)
        np.testing.assert_array_equal(result, output)
