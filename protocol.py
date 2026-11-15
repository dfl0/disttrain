"""Sync/async wire formats and parameter-only tensor serialization.

DTR1 packets carry unsigned 64-bit version, sample count and tensor length in
network byte order, followed by little-endian float32 values. One root update
per round makes the root version also the round ID. Restarts/retries are unsupported.
"""
import struct
import numpy as np
from enum import IntEnum


class Role(IntEnum):
    ROOT_PS = 1
    EDGE_PS = 2
    WORKER = 3
    EVAL = 4


class Msg(IntEnum):
    PUSH_PARAMS = 1
    PUSH_GRADS = 2
    STOP = 3
    EVAL = 4
    REGISTER = 5
    ASSIGN = 6
    DONE = 7
    ASYNC_GRADS = 8
    ASYNC_PARAMS = 9
    ASYNC_DONE = 10


_HEADER = struct.Struct('!4sQQQ')
_ASSIGN = struct.Struct('!q')  # global worker rank; -1 identifies an edge
_VERSION = struct.Struct('!Q')
MAX_VERSION = 2**64 - 1


def encode_packet(version, values, sample_count=0):
    if (type(version) is not int or not 0 <= version <= MAX_VERSION
            or type(sample_count) is not int or not 0 <= sample_count <= MAX_VERSION):
        raise ValueError('version and sample count must be uint64 integers')
    tensor = _tensor_bytes(values)
    return _HEADER.pack(b'DTR1', version, sample_count, values.size) + tensor


def _tensor_bytes(values):
    if (not isinstance(values, np.ndarray) or values.ndim != 1
            or values.dtype != np.dtype(np.float32) or not values.flags.c_contiguous
            or not np.isfinite(values).all()):
        raise ValueError('packet values must be finite contiguous float32 vector')
    return values.astype('<f4', copy=False).tobytes()


def decode_packet(payload, n_params, *, gradients=False):
    if len(payload) < _HEADER.size:
        raise ValueError('truncated packet header')
    magic, version, sample_count, length = _HEADER.unpack_from(payload)
    if magic != b'DTR1' or length != n_params or len(payload) != _HEADER.size + length * 4:
        raise ValueError('unsupported packet schema or incorrect tensor length')
    if (gradients and sample_count == 0) or (not gradients and sample_count != 0):
        raise ValueError('gradients require a positive sample count; snapshots require zero')
    values = np.frombuffer(payload, dtype='<f4', offset=_HEADER.size).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('non-finite packet values')
    return version, sample_count, values


def encode_assignment(index):
    if type(index) is not int or index < -1:
        raise ValueError('assignment must be a worker rank or -1 for an edge')
    return _ASSIGN.pack(index)


def decode_assignment(payload):
    if len(payload) != _ASSIGN.size:
        raise ValueError('invalid assignment length')
    index = _ASSIGN.unpack(payload)[0]
    if index < -1:
        raise ValueError('invalid assignment rank')
    return index


def encode_version(version):
    if type(version) is not int or not 0 <= version <= MAX_VERSION:
        raise ValueError('version must be a uint64 integer')
    return _VERSION.pack(version)


def decode_version(payload):
    if len(payload) != _VERSION.size:
        raise ValueError('invalid completion version')
    return _VERSION.unpack(payload)[0]


# preserve each worker contribution when summing gradients
_ASYNC_HEADER = struct.Struct('!4sQQQ')
_RECORD = struct.Struct('!QQQQ')  # rank, contribution id, source version, samples


def _validate_records(records):
    ranks = set()
    count = 0
    if not records:
        raise ValueError('async aggregate requires contribution records')
    for record in records:
        if len(record) != 4 or any(type(v) is not int or not 0 <= v <= MAX_VERSION for v in record):
            raise ValueError('invalid async contribution record')
        rank, contribution, version, samples = record
        if rank in ranks or contribution == 0 or samples == 0:
            raise ValueError('duplicate rank or zero contribution/sample count')
        ranks.add(rank)
        count += samples
    if count > MAX_VERSION:
        raise ValueError('async sample-count overflow')


def encode_async_gradients(request_id, records, values):
    encode_version(request_id)
    if request_id == 0:
        raise ValueError('async request IDs start at one')
    _validate_records(records)
    tensor = _tensor_bytes(values)
    return (_ASYNC_HEADER.pack(b'DTA1', request_id, len(records), values.size)
            + b''.join(_RECORD.pack(*record) for record in records) + tensor)


def decode_async_gradients(payload, n_params):
    if len(payload) < _ASYNC_HEADER.size:
        raise ValueError('truncated async header')
    magic, request_id, count, length = _ASYNC_HEADER.unpack_from(payload)
    offset = _ASYNC_HEADER.size + count * _RECORD.size
    if magic != b'DTA1' or request_id == 0 or length != n_params or len(payload) != offset + length * 4:
        raise ValueError('invalid async gradient packet')
    records = [_RECORD.unpack_from(payload, _ASYNC_HEADER.size + i * _RECORD.size) for i in range(count)]
    _validate_records(records)
    values = np.frombuffer(payload, dtype='<f4', offset=offset).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError('non-finite async gradients')
    return request_id, records, values


def encode_async_params(request_id, version, values):
    return encode_version(request_id) + encode_packet(version, values)


def decode_async_params(payload, n_params):
    request_id = decode_version(payload[:_VERSION.size])
    version, _, values = decode_packet(payload[_VERSION.size:], n_params)
    return request_id, version, values


_EVAL_METADATA = struct.Struct('!Qd')


def encode_evaluation(version, samples, elapsed, values):
    encode_version(samples)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError('invalid evaluation snapshot time')
    return _EVAL_METADATA.pack(samples, elapsed) + encode_packet(version, values)


def decode_evaluation(payload, n_params):
    if len(payload) < _EVAL_METADATA.size:
        raise ValueError('truncated evaluation metadata')
    samples, elapsed = _EVAL_METADATA.unpack_from(payload)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError('invalid evaluation snapshot time')
    version, _, params = decode_packet(payload[_EVAL_METADATA.size:], n_params)
    return version, samples, elapsed, params


class ParamSerializer:
    """Parameter-only dense contiguous CPU/float32 serialization.

    Offsets are cached, but identity/layout/buffers are checked before every
    operation. This deliberately trades a parameter walk for fail-fast safety.
    Gradients are checked only when flattening gradients, not when moving parameters.
    No synchronized model buffers or implicit dtype/device conversions exist.
    """

    def __init__(self, model):
        import torch
        self._torch = torch
        self._model = model
        self._layout = []
        offset = 0
        for name, parameter in model.named_parameters():
            shape = tuple(parameter.shape)
            n = parameter.numel()
            self._layout.append((name, parameter, shape, offset, n))
            offset += n
        self.n_params = offset
        self._preflight()

    def _tensor_contract(self, name, tensor, shape):
        expected = f"shape={shape}, device=cpu, dtype=float32, layout=strided, contiguous"
        actual = (f"shape={tuple(tensor.shape)}, device={tensor.device}, "
                  f"dtype={tensor.dtype}, layout={tensor.layout}, "
                  f"contiguous={tensor.is_contiguous() if tensor.layout == self._torch.strided else False}")
        if (tuple(tensor.shape) != shape or tensor.device.type != "cpu"
                or tensor.dtype != self._torch.float32
                or tensor.layout != self._torch.strided or not tensor.is_contiguous()):
            raise ValueError(f"{name}: expected {expected}; actual {actual}")

    def _buffer_contract(self):
        for module_name, module in self._model.named_modules():
            for name, buffer in module._buffers.items():
                if buffer is None:
                    continue
                inert_bn_counter = (
                    type(module) is self._torch.nn.BatchNorm2d
                    and name == "num_batches_tracked"
                    and not module.track_running_stats
                    and module.running_mean is None and module.running_var is None
                )
                if not inert_bn_counter:
                    full_name = f"{module_name}.{name}" if module_name else name
                    raise ValueError(f"buffer {full_name}: expected parameter-only model "
                                     f"(no active registered buffers); actual {buffer}")

    def _preflight(self, *, gradients=False):
        current = list(self._model.named_parameters())
        if len(current) != len(self._layout):
            raise ValueError("parameter layout changed: expected names "
                             f"{[entry[0] for entry in self._layout]}; "
                             f"actual {[name for name, _ in current]}")
        for (name, parameter), (cached_name, cached, shape, _, _) in zip(current, self._layout):
            if name != cached_name or parameter is not cached:
                raise ValueError(f"parameter {name}: expected original object/order for "
                                 f"{cached_name}; actual replacement or reordered parameter")
            self._tensor_contract(f"parameter {name}", parameter, shape)
            if gradients and parameter.grad is not None:
                self._tensor_contract(f"gradient {name}", parameter.grad, shape)
        self._buffer_contract()

    def _array_contract(self, array, *, writable):
        if (not isinstance(array, np.ndarray) or array.ndim != 1
                or array.shape != (self.n_params,)
                or array.dtype != np.dtype(np.float32) or not array.dtype.isnative
                or not array.flags.c_contiguous or (writable and not array.flags.writeable)):
            raise ValueError(f"flat array: expected 1D length {self.n_params}, native float32, "
                             f"C-contiguous{' writable' if writable else ''}; "
                             f"actual shape={getattr(array, 'shape', None)}, "
                             f"dtype={getattr(array, 'dtype', None)}, "
                             f"flags={getattr(array, 'flags', None)}")

    def flatten_params_into(self, out):
        self._preflight()
        self._array_contract(out, writable=True)
        for _, parameter, _, off, n in self._layout:
            out[off:off + n] = parameter.data.view(-1).detach().numpy()

    def flatten_grads_into(self, out):
        self._preflight(gradients=True)
        self._array_contract(out, writable=True)
        for _, parameter, _, off, n in self._layout:
            if parameter.grad is None:
                out[off:off + n] = 0.0
            else:
                out[off:off + n] = parameter.grad.view(-1).detach().numpy()

    def load_params(self, flat):
        self._preflight()
        self._array_contract(flat, writable=False)
        # PyTorch does not support writing through read-only NumPy storage.
        source = flat if flat.flags.writeable else flat.copy()
        flat_t = self._torch.from_numpy(source)
        for _, parameter, _, off, n in self._layout:
            parameter.data.copy_(flat_t[off:off + n].view_as(parameter))
