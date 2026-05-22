# Distributed Hierarchical Training Framework

Research-oriented distributed deep learning framework built with PyTorch, MPI, and explicit parameter-server communication.

Designed for experimenting with:
* hierarchical aggregation
* asynchronous training
* communication bottlenecks
* heterogeneous compute/network conditions
* resource-constrained devices
* distributed systems observability

The implementation intentionally avoids high-level distributed abstractions. Communication is explicit and uses:
* non-blocking MPI sends/receives
* manual parameter serialization
* explicit message types
* custom synchronization logic

## Features

* Flat and hierarchical parameter-server topologies
* Synchronous and asynchronous aggregation
* Independently configurable root/edge synchronization
* Bandwidth throttling using `tc`
* CPU throttling using Linux cgroups
* Runtime metrics collection
* Timing instrumentation
* MPI-based distributed execution

### Topologies

Flat:
```text
Root Parameter Server
├── Worker
├── Worker
├── Worker
└── ...
```

Hierarchical:
```text
Root Parameter Server
├── Edge Parameter Server
│   ├── Worker
│   ├── Worker
│   └── ...
├── Edge Parameter Server
│   ├── Worker
│   ├── Worker
│   └── ...
├── ...
└── Evaluator
```

## Requirements

Controller node:
* Linux
* OpenMPI
* Python 3.10+
    - mpi4py
* SSH access to all nodes

Training nodes:
* Linux
* OpenMPI
* Python 3.10+
    - mpi4py
    - PyTorch
    - torchvision
    - NumPy
    - PyYAML
    - psutil

## Installation

Clone the repository on each device, then in the project root:
```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements_<controller|node>.txt
```

## Repository Layout

```text
.
├── dist_train.py
├── timetree.py
├── run.sh
├── multirun.sh
├── hosts
├── configs/
└── logs/
```

## Hosts File

`hosts` contains one hostname/IP per line:

```text
node1
node2
node3
node4
```

One extra node is required for the evaluator.

Passwordless SSH is recommended:

```bash
ssh-copy-id user@node1
...
```

## Configuration

Training topology is defined in YAML.

Example:

```yaml
sync: false  # root PS

edge_servers:
  - sync: true  # edge PS 1
    workers:
      - cpu: 100
      - cpu: 100

  - sync: true  # edge PS 2
    workers:
      - cpu: 30
        bw: 10mbit
      - cpu: 30
        bw: 10mbit
```

Worker options:
* `cpu`: CPU limit percentage
* `bw`: bandwidth limit

Each individual root and edge server supports:
* `sync: true`
* `sync: false`

This allows:
* fully synchronous/asynchronous
* mixed synchronization strategies

## Running

Single experiment:

```bash
./run.sh <path-to-config> <run-name>
```

Multiple experiments:

```bash
./multirun.sh
```

Logs are written to:

```text
logs/
├── <run-name>_all.log
├── <run-name>_eval.log
└── <run-name>_metrics.log
```

## Runtime Metrics

The framework collects:
* CPU usage
* CPU frequency
* memory usage
* thermal throttling
* network throughput
* MPI send queue state
* timing breakdowns

## Datasets

Datasets are expected at:

```text
/mnt/data
```

Automatic downloading is disabled. For now, just enable automatic downloading for the first run with a new dataset.

## Notes

Current limitations:
- Linux-only resource throttling
- no checkpointing
- no fault tolerance
- no authentication/security
- manual serialization overhead

This project is intended for distributed systems and training research, not production deployment.

## License

TBD
