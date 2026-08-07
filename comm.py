from enum import IntEnum


class Role(IntEnum):
    ROOT_PS = 1
    EDGE_PS = 2
    WORKER  = 3
    EVAL    = 4


class Msg(IntEnum):
    PUSH_PARAMS = 1
    PUSH_GRADS  = 2
    STOP        = 3
    EVAL        = 4
    REGISTER    = 5   # child → PS on first connect; payload = hostname bytes
    ASSIGN      = 6   # PS → child; payload = int32(worker_idx)
