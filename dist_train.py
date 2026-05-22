import time
import os
import socket
import numpy as np
import torch
import torch.nn as nn
import torchvision
import subprocess
import yaml
import shlex
import threading
import psutil
import warnings
from mpi4py import MPI
from enum import IntEnum
from collections import defaultdict
from contextlib import contextmanager

from timetree import TimeTree


warnings.filterwarnings("ignore", module="torchvision.datasets.cifar")


# ----  set training options  ------------------------------------------------+

n_epochs = 1
batch_size = 32
learning_rate = 0.01


# ----  get node/proc info  --------------------------------------------------+

hostname = socket.gethostname()
world = MPI.COMM_WORLD
world_rank = world.Get_rank()
world_size = world.Get_size()
config_file = 'config.yaml'

device_role = None

world.Barrier()
global_start_time = time.perf_counter()


global_state = {
    "sections": [],
    "n_parent_send_reqs": 0,
    "n_child_send_reqs": 0,
    "send_reqs_children": []
}


# ----  define model  --------------------------------------------------------+

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        return self.model(x)


# LeNet-based architecture adapted to fit CIFAR10 as input data
class LeNetCIFAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2, 2),

            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2, 2),

            torch.nn.Flatten(),
            torch.nn.Linear(16*5*5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)


# AlexNet-based architecture adapted to fit CIFAR10 as input data
class AlexNetCIFAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(),

            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(),

            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Flatten(),

            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 10)
        )

    def forward(self, x):
        return self.model(x)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet20CIFAR(torch.nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet20CIFAR, self).__init__()
        self.inplanes = 16
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# ----  utility classes  -----------------------------------------------------+

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


class Timer:
    def __init__(self):
        self.times = defaultdict(list)

    @contextmanager
    def measure(self, category):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.times[category].append(elapsed)

    def get_times(self, category):
        return self.times[category] if category in self.times else self.times

    def get_stats(self, category):
        if category not in self.times or not self.times[category]:
            return {"count": 0, "total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

        durations = self.times[category]
        return {
            "count": len(durations),
            "total": sum(durations),
            "mean": sum(durations)/len(durations),
            "min": min(durations),
            "max": max(durations),
        }

    def reset(self):
        self.times.clear()
        # for category in self.times:
        #     self.times[category] = [0]

    def summary(self, categories=None):
        if categories is None:
            categories = self.times.keys()

        summary = {}

        for c in categories:
            stats = self.get_stats(c)
            if stats.get("count") > 0:
                summary[c] = {**stats}

        return summary


# ----  utility functions  ---------------------------------------------------+

def build_topology_from_yaml(path, n_nodes):
    # use user defined yaml file, that specifies structure and bottlenecks, but not IDs or hostnames
    with open(path, "r") as f:
        root = yaml.safe_load(f)

    parents = {0: None}
    children = {i: [] for i in range(n_nodes)}
    central_server = 0
    id_counter = 1
    edge_servers = []
    workers = []
    specs = {}
    sync = {0: root.get("sync")}

    if "edge_servers" in root:
        for child in root.get("edge_servers"):
            child_id = id_counter
            id_counter += 1
            children[central_server].append(child_id)
            parents[child_id] = central_server
            edge_servers.append(child_id)
            sync[child_id] = child.get("sync")
            for worker in child.get("workers", []):
                worker_id = id_counter
                id_counter += 1
                workers.append(worker_id)
                specs[worker_id] = [worker.get("bw"), worker.get("cpu")]
                children[child_id].append(worker_id)
                parents[worker_id] = child_id
    elif "workers" in root:
        for worker in root.get("workers"):
            worker_id = id_counter
            id_counter += 1
            workers.append(worker_id)
            children[central_server].append(worker_id)
            parents[worker_id] = central_server
            specs[worker_id] = [worker.get("bw"), worker.get("cpu")]

    return {
        "central_server": central_server,
        "edge_servers": edge_servers,
        "workers": workers,
        "parents": parents,
        "children": children,
        "sync": sync,
        "bw_cpu_limits": specs,
    }


def run_cmd(cmd, check=False, capture_output=True):
    result = subprocess.run(
        shlex.split(cmd),
        check=check,
        capture_output=capture_output,
        text=True
    )
    return result.stdout.strip()


def apply_bw_cpu_limits(bw_cpu_limits):
    # *4000 here because we have 4 cores
    #log(f"{os.getpid()}")
    if bw_cpu_limits[0]:
        run_cmd("tc qdisc add dev eth0 root handle 1: htb default 1")
        run_cmd(f"tc class add dev eth0 parent 1: classid 1:1 htb rate {bw_cpu_limits[0]}")
        # setup a virtual device through which to redirect incoming traffic
        run_cmd("modprobe ifb numifbs=1")
        run_cmd("ip link set dev ifb0 up")
        run_cmd("tc qdisc add dev eth0 handle ffff: ingress")
        run_cmd("tc filter add dev eth0 parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0")
        run_cmd("tc qdisc add dev ifb0 root handle 2: htb default 1")
        run_cmd(f"tc class add dev ifb0 parent 2: classid 2:1 htb rate {bw_cpu_limits[0]}")

    if bw_cpu_limits[1]:
        cpu_limit = bw_cpu_limits[1] * 4000
        run_cmd("mkdir -p /sys/fs/cgroup/train_cpu")
        with open("/sys/fs/cgroup/train_cpu/cgroup.procs", "w") as f:
            f.write(str(os.getpid()))
        with open("/sys/fs/cgroup/train_cpu/cpu.max", "w") as f:
            f.write(str(f"{cpu_limit} 100000"))

def remove_bw_cpu_limits():
    # remove process from the cgroup, then rm the directory to clean up, delete ifb and tc rules
    run_cmd("tc qdisc del root dev eth0")
    run_cmd("tc qdisc del dev eth0 handle ffff: ingress")
    run_cmd("tc qdisc del root dev ifb0")
    run_cmd("ip link set dev ifb0 down")
    run_cmd("modprobe -r ifb")

    run_cmd("echo 0 > /sys/fs/cgroup/train_cpu/cgroup.procs")
    run_cmd("rmdir /sys/fs/cgroup/train_cpu")


def build_topology(n_nodes, hierarchical=False, k=None):
    assert n_nodes >= 1

    parents = {}                                # maintain map from rank -> parent
    children = {i: [] for i in range(n_nodes)}  # similarly, from rank -> children

    central_server = 0
    parents[central_server] = None

    # -- flat topology --

    if not hierarchical:
       workers = list(range(1, n_nodes))  # since for now, we are just using rank 0 as central PS

       for w in workers:
           parents[w] = central_server
           children[central_server].append(w)

       return {"central_server": central_server,
               "workers": workers,
               "parents": parents,
               "children": children}

    # -- hierarchical topology --

    if k is None:
        k = max(1, int((n_nodes - 1)**0.5))  # choose number of clusters automatically if custom number not passed

    edge_servers = list(range(1, min(k + 1, n_nodes)))
    workers = list(range(1 + len(edge_servers), n_nodes))

    for e in edge_servers:
        parents[e] = central_server
        children[central_server].append(e)

    for i, w in enumerate(workers):
        e = edge_servers[i % len(edge_servers)]
        parents[w] = e
        children[e].append(w)

    return {"central_server": central_server,
            "edge_servers": edge_servers,
            "workers": workers,
            "parents": parents,
            "children": children}


def get_role(rank, topology):
    role = None

    if rank == topology.get("central_server"):
        role = Role.ROOT_PS
    elif "edge_servers" in topology and rank in topology.get("edge_servers"):
        role = Role.EDGE_PS
    elif rank in topology.get("workers"):
        role = Role.WORKER
    else:
        return Role.EVAL, None, None, None

    return role, topology.get("parents")[rank], topology.get("children")[rank], topology.get("sync").get(rank, None)


def log(msg, indent=0):
    t = time.perf_counter() - global_start_time
    whole = int(t)
    ms = int((t - whole) * 1000)
    timestamp = time.strftime("%H:%M:%S", time.gmtime(whole))
    timestamp = f"{timestamp}:{ms:03d}"

    global device_role
    assert device_role

    prefix = (
        f"({timestamp}) "
        f"{str(world_rank):>{len(str(world_size - 1))}} "
        f"[{device_role.name:<{8}}]  "
    )

    indent_str = "    "*indent

    print(f"{prefix}{indent_str}{msg}", flush=True)


def flatten_params(model):
    params = [p.data.view(-1) for p in model.parameters()]
    flat = torch.cat(params)
    if flat.dtype != torch.float32:
        flat = flat.float()
    return flat.numpy()
    # return flat.detach().cpu().numpy().astype(np.float32)


def load_params(model, flat_params):
    flat_params_torch = torch.from_numpy(flat_params)
    offset = 0
    for p in model.parameters():
        n = p.numel()
        # p.data.copy_(
        #     torch.from_numpy(flat_params[offset:(offset + n)])
        #     .view_as(p)
        # )
        p.data.copy_(
            flat_params_torch[offset:(offset + n)]
            .view_as(p)
            # .to(device=p.device, dtype=p.dtype)
        )
        offset += n


def flatten_grads(model):
    # grads = [p.grad.view(-1) for p in model.parameters()]
    # flat = torch.cat(grads)
    # return flat.detach().cpu().numpy().astype(np.float32)

    grads = [torch.zeros_like(p).view(-1) if p.grad is None else
             p.grad.view(-1)
             for p in model.parameters()]
    flat = torch.cat(grads)
    flat = flat.detach().cpu()
    if flat.dtype != torch.float32:
        flat = flat.float()
    return flat.numpy()


def load_grads(model, flat_grads):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        grad_np = flat_grads[offset:(offset + n)].reshape(p.shape)
        grad_t = torch.from_numpy(grad_np)
        if p.grad is None:
            p.grad = grad_t.clone()
        else:
            p.grad.copy_(grad_t)
        offset += n


def init_params(m):
    if type(m) is torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def disable_state(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.track_running_stats = False
        m.running_mean = None
        m.running_var = None


def load_mnist():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = torchvision.datasets.MNIST(
        root="/mnt/data",
        train=True,
        download=False,
        transform=transform
    )

    test_ds = torchvision.datasets.MNIST(
        root="/mnt/data",
        train=False,
        download=False,
        transform=transform
    )

    return train_ds, test_ds


def load_cifar10():
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root="/mnt/data",
        train=True,
        download=False,
        transform=transform_train
    )

    test_ds = torchvision.datasets.CIFAR10(
        root="/mnt/data",
        train=False,
        download=False,
        transform=transform_test
    )

    return train_ds, test_ds


def cleanup_mpi_req(req):
    req.Cancel()
    try:
        req.Cancel()
    except MPI.Exception:
        pass


def cancel_recv_req(req):
    if req is not None and req != MPI.REQUEST_NULL:
        req.Cancel()
        try:
            MPI.Request.Wait(req)
        except MPI.Exception:
            pass


def safe_wait_send_req(req, timeout=1.0):
    if req is not None and req != MPI.REQUEST_NULL:
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout:
            if req.Test():
                return
            time.sleep(0.01)

        # forcefully cancel if request times out
        req.Cancel()


def collect_sys_metrics(collect_metrics, interval=0.04, quiet=False):
    collect_metrics.wait()

    def get_net_bytes():
        counter = psutil.net_io_counters(pernic=True)["eth0"]
        return (counter.bytes_sent, counter.bytes_recv)

    t0 = time.perf_counter()
    tot_bytes = get_net_bytes()
    while collect_metrics.is_set():
        last_tot_bytes = tot_bytes

        # cpu = psutil.cpu_percent(min(interval, 0.1))
        cpu = psutil.cpu_percent()  # TODO: is an interval needed here?
        freq = psutil.cpu_freq().current
        mem = psutil.virtual_memory().percent
        temp = psutil.sensors_temperatures()["cpu_thermal"][0].current

        throttled_raw = run_cmd("vcgencmd get_throttled")
        throttled_val = int(throttled_raw.split('=')[-1], 16)
        throttled_now = (throttled_val >> 2) % 2   # if the pi is currently throttled
        throttled_bef = (throttled_val >> 18) % 2  # if throttling has occured before

        t1 = time.perf_counter()
        tot_bytes = get_net_bytes()
        t_delta = t1 - t0
        t0 = time.perf_counter()

        bytes_sent = tot_bytes[0] - last_tot_bytes[0]
        bytes_received = tot_bytes[1] - last_tot_bytes[1]
        ul_Bps = (bytes_sent)/(t_delta)
        dl_Bps = (bytes_received)/(t_delta)

        global global_state
        section, start = global_state["sections"][-1] if global_state["sections"] else (None, None)

        n_parent_send_reqs = global_state["n_parent_send_reqs"]
        n_child_send_reqs = global_state["n_child_send_reqs"]
        send_reqs_children = global_state["send_reqs_children"]

        now = time.perf_counter() - global_start_time
        time.sleep(interval - (now % interval))
        # time.sleep(interval - 1)
        now = time.perf_counter() - global_start_time

        if not quiet:
            # time.sleep(world_rank*min(interval/world_size, 0.01))
            time.sleep(world_rank*interval/world_size)
            log(
                "[METRICS]  "
                f"t: {now} \t"
                f"cpu: {cpu}% (freq={freq}) \t"
                f"mem: {mem}% \t"
                f"temp: {temp} \t"
                f"throttled: {throttled_now} (now), {throttled_bef} (before)\t"
                f"net ul: {ul_Bps/1000.0:.4f} kBps \t"
                f"net dl: {dl_Bps/1000.0:.4f} kBps \t"
                f"section: {section} ({start - global_start_time if start else None}) \t"
                f"send requests: {n_parent_send_reqs} (parent), {n_child_send_reqs} (child) {send_reqs_children}"
            )


# ----  role functions  ------------------------------------------------------+

def run_evaluator(model, root, comm):
    log("[EVALUATOR]")

    # -- load test data  --
    # _, test_ds = load_mnist()
    _, test_ds = load_cifar10()

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        drop_last=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    global global_state
    timer = TimeTree(print_func=log, category_stack_ref=global_state["sections"])

    flat_params = flatten_params(model)
    packet_size = flat_params.size + 1

    test_acc = 0
    model_version = 0

    accuracy_threshold = 0.90
    # time_threshold = 1*60*60
    time_threshold = 30

    recv_buf = np.empty(packet_size, dtype=np.float32)
    new_params = False

    start_time = time.perf_counter()
    while True:
        # -- drain all parameter pushes from root to get newest version --
        with timer.measure("drain"):
            drained = 0
            while comm.Iprobe(source=root, tag=Msg.EVAL):
                comm.Recv(recv_buf, source=root, tag=Msg.EVAL)

                drained += 1
                new_version = int(recv_buf[0])

                if new_version > model_version:
                    model_version = new_version
                    flat_params[:] = recv_buf[1:]
                    new_params = True
                else:
                    assert False

        if new_params:
            recv_time = time.perf_counter() - start_time
            log(f"drained {drained} pushed params")
            log(f"Newest version received: {model_version}. Starting evaluation...")

            with timer.measure("deserialize_params"):
                load_params(model, flat_params)
            new_params = False

            total_loss = 0.0
            total_samples = 0
            total_correct = 0

            with timer.measure("eval"):
                model.eval()
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        y_pred = model(X_batch)

                        predicted = torch.argmax(y_pred, 1)
                        total_correct += (predicted == y_batch).sum().item()

                        loss = loss_fn(y_pred, y_batch)
                        test_batch_size = X_batch.size(0)

                        total_loss += loss.item()*test_batch_size
                        total_samples += test_batch_size

                test_acc = total_correct/total_samples
                test_loss = total_loss/total_samples

            done_time = time.perf_counter() - start_time
            log('*'*100)
            log(f"Evaluation (recv'd after {recv_time:.2f}s, done after {done_time:.2f}s): version {model_version}, accuracy: {test_acc:.2%}, loss: {test_loss:.4f}")
            log('*'*100)

            if test_acc >= accuracy_threshold or done_time >= time_threshold:
            # if True:
                log("sending STOP message...")
                comm.Send(np.array([0], dtype=np.int32), dest=0, tag=Msg.STOP)
                break

        else:
            # log(f"No updated parameters received")
            time.sleep(0.5)

    log("evaluator terminating...")
    log("evaluator done")


def run_parameter_server(model, children, evaluator, comm, sync=True):
    rank = comm.Get_rank()
    assert rank == 0

    log(f"[{'SYNC' if sync else 'ASYNC'} PS]")

    # -- initialize optimizer --
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    global global_state
    timer = TimeTree(print_func=log, category_stack_ref=global_state["sections"])

    # -- initialize global params --
    model.apply(init_params)
    model_version = 0

    flat_params = flatten_params(model)

    # -- pre-allocate buffers --
    packet_size = flat_params.size + 1
    child_packets = [
        np.empty(packet_size, dtype=np.float32)  # maintain separate buffers to receive from each child
        for _ in children
    ]
    send_packet = np.empty(packet_size, dtype=np.float32)
    stop_buf = np.empty(1, dtype=np.int32)

    # -- pre-post receives for grads pushed from children --
    child_recv_reqs = [
        comm.Irecv(child_packets[i], source=children[i], tag=Msg.PUSH_GRADS)
        for i in range(len(children))
    ]

    send_reqs = []
    stop_reqs = []

    # -- push initial params --
    # log("sending initial params")
    send_packet[0] = model_version
    send_packet[1:] = flat_params

    for child in children:
        dst = child
        buf = send_packet.copy()
        req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_PARAMS)
        send_reqs.append((req, buf, dst))

    stop_recv_req = comm.Irecv(stop_buf, source=evaluator, tag=Msg.STOP)

    aggr_grads_buf = np.zeros(flat_params.size, dtype=np.float32)
    total_weight = 0.0
    n_updates = 0

    # n_to_wait = max(1, len(children)//2)
    n_to_wait = lambda: max(1, active_children//2)
    eval_interval = 200

    active_children = len(children)
    children_done = set()  # keep track of children that have sent an update if synchronous
    status = MPI.Status()

    log("Starting training...")

    timer.reset()
    with timer.measure("train"):

        while active_children > 0:
            reqs = child_recv_reqs + [stop_recv_req]
            with timer.measure("wait") as t:
                idx = MPI.Request.Waitany(reqs)

                if idx == MPI.UNDEFINED:
                    log("UNDEFINED INDEX (this should not happen)")
                    break

                if idx == len(reqs) - 1:
                    t["tag"] = "stop"
                    log("Stop signal received, propagating to children")
                    for child in children:
                        buf = np.array([0], dtype=np.int32)
                        req = comm.Isend(buf, dest=child, tag=Msg.STOP)
                        stop_reqs.append((req, buf))
                    break
                else:
                    t["tag"] = children[idx]

            # otherwise, idx is one of child_recv_reqs

            src = children[idx]
            packet = child_packets[idx]

            params_version = int(packet[0])
            params_delta = packet[1:]
            # params_delta = flat_params - params

            # eta = 0.1
            # eta = 1/len(children)
            # flat_params = (1 - eta)*flat_params + eta*params
            # flat_params -= learning_rate*params_delta
            flat_params -= params_delta/active_children
            model_version += 1

            child_recv_reqs[idx] = comm.Irecv(child_packets[idx], source=src, tag=Msg.PUSH_GRADS)

            send_packet[0] = model_version
            send_packet[1:] = flat_params

            if sync:
                children_done.add(src)
                # log(f"children done: {children_done}")
                # -- send updated parameters back to children after receiving update from each --
                if len(children_done) == active_children:
                    # log("all children done, sending back updated model")
                    for child in children_done:
                        # with timer.measure("push_params") as t:  # TODO: good way to time actual sending time?
                        dst = child 
                        buf = send_packet.copy()
                        req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_PARAMS)
                        send_reqs.append((req, buf, dst))
                        t["tag"] = child
                    children_done.clear()

            else:
                # -- send updated parameters back to child --
                # with timer.measure("push_params") as t:
                dst = src
                buf = send_packet.copy() 
                req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_PARAMS)
                send_reqs.append((req, buf, dst))
                t["tag"] = src

            if model_version % eval_interval == 0:
            # if model_version == 1:
                log(f"sending params to evaluator, version: {model_version}")
                # with timer.measure("push_params") as t:
                dst = evaluator
                buf = send_packet.copy()
                req = comm.Isend(buf, dest=dst, tag=Msg.EVAL)
                send_reqs.append((req, buf, dst))
                t["tag"] = "eval"

            # -- clean up finished sends --
            global_state["send_reqs_children"] = []
            for i in reversed(range(len(send_reqs))):
                req, buf, dst = send_reqs[i]
                if req.Test():
                    send_reqs.pop(i)
                else:
                    global_state["send_reqs_children"].append(dst)

            global_state["n_child_send_reqs"] = len(send_reqs)

    log("root ps terminating...")

    log(f"waiting to send {len(stop_reqs)} stop signals...")
    for req, _ in stop_reqs:
        safe_wait_send_req(req, timeout=2.0)

    for child in children:
        while comm.Iprobe(source=child, tag=Msg.PUSH_GRADS):
            comm.Recv(MPI.IN_PLACE, source=child, tag=Msg.PUSH_GRADS)

    log("cancelling child recv reqs")
    for req in child_recv_reqs:
        cancel_recv_req(req)

    log("safe waiting send reqs")
    for req, _, _ in send_reqs:
        safe_wait_send_req(req, timeout=0.5)
        log("...")

    log("root ps done cleaning up")

    timer.print_tree()


def run_edge_parameter_server(model, parent, children, comm, sync=True):

    log(f"[{'SYNC' if sync else 'ASYNC'} EDGE PS] parent: {parent}")

    global global_state
    timer = TimeTree(print_func=log, category_stack_ref=global_state["sections"])

    flat_params = flatten_params(model)

    # -- pre-allocate buffers --
    packet_size = flat_params.size + 1
    parent_packet = np.empty(packet_size, dtype=np.float32)
    child_packets = [
        np.empty(packet_size, dtype=np.float32)  # maintain separate buffers to receive from each child
        for _ in children
    ]
    send_packet = np.empty(packet_size, dtype=np.float32)
    stop_buf = np.empty(1, dtype=np.int32)

    # -- pre-post receives for grads pushed from workers --
    child_recv_reqs = [
        comm.Irecv(child_packets[i], source=children[i], tag=Msg.PUSH_GRADS)
        for i in range(len(children))
    ]

    parent_send_reqs = []
    child_send_reqs = []
    stop_reqs= []

    # -- wait for initial params --
    comm.Recv(parent_packet, source=parent, tag=Msg.PUSH_PARAMS)
    model_version = int(parent_packet[0])
    flat_params[:] = parent_packet[1:]
    anchor_params = flat_params.copy()
    # anchor_params = None
    # model_version = -1

    # -- push initial params to workers --
    # log("sending out initial params")
    for child in children:
        dst = child
        buf = parent_packet.copy()
        req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_PARAMS)
        child_send_reqs.append((req, buf, dst))

    parent_recv_req = comm.Irecv(parent_packet, source=parent, tag=Msg.PUSH_PARAMS)

    # -- pre-post receive for stop request from parent --
    stop_recv_req = comm.Irecv(stop_buf, source=parent, tag=Msg.STOP)

    aggr_grads_buf = np.zeros(flat_params.size, dtype=np.float32)
    total_weight = 0.0  # accumulate weights applied to incoming grads for normalization
    grads_versions = []
    n_updates = 0

    active_children = len(children)
    children_done = set()  # keep track of children that have sent an update if synchronous
    stop_flag = False
    new_params = False
    status = MPI.Status()

    n_to_wait = lambda: max(1, active_children//2)
    # n_to_wait = lambda: active_children

    timer.reset()
    with timer.measure("train"):
        while active_children > 0:
            reqs = child_recv_reqs + [parent_recv_req, stop_recv_req]
            with timer.measure("wait") as t:
                idx = MPI.Request.Waitany(reqs)

                if idx == MPI.UNDEFINED:
                    log("UNDEFINED INDEX (this should not happen)")
                    break
                elif idx == len(reqs) - 1:
                    t["tag"] = "stop"
                elif idx == len(reqs) - 2:
                    t["tag"] = parent
                else:
                    t["tag"] = children[idx]

            if idx == len(reqs) - 1:
                log("Stop signal received, propagating to children")
                for child in children:
                    buf = np.array([0], dtype=np.int32)
                    req = comm.Isend(buf, dest=child, tag=Msg.STOP)
                    stop_reqs.append((req, buf))
                break

            elif idx == len(reqs) - 2:
                model_version = int(parent_packet[0])
                flat_params[:] = parent_packet[1:]
                anchor_params = flat_params.copy()

                if sync:
                    for child in children:
                        # with timer.measure("push_params") as t:
                        dst = child
                        buf = parent_packet.copy()
                        req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_PARAMS)
                        child_send_reqs.append((req, buf, dst))
                        t["tag"] = child

                parent_recv_req = comm.Irecv(parent_packet, source=parent, tag=Msg.PUSH_PARAMS)

            else:
                src = children[idx]
                packet = child_packets[idx]

                grads_version = int(packet[0])
                grads = packet[1:]

                # eta = 0.1
                # eta = 1/len(children)
                # flat_params = (1 - eta)*flat_params + eta*params
                flat_params -= learning_rate*grads
                n_updates += 1

                child_recv_reqs[idx] = comm.Irecv(child_packets[idx], source=src, tag=Msg.PUSH_GRADS)

                if sync:
                    children_done.add(src)

                    if len(children_done) == active_children:
                        # -- send local parameter deltas to root after receiving update from each worker --
                        # log("received update from each worker, sending up local model")
                        send_packet[0] = model_version
                        send_packet[1:] = anchor_params - flat_params  # send parameter deltas
                        # with timer.measure("push_params") as t:
                        dst = parent
                        buf = send_packet.copy()
                        req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_GRADS)
                        parent_send_reqs.append((req, buf, dst))
                        t["tag"] = parent
                        children_done.clear()

                else:
                    # -- send updated parameters back to worker --
                    # log("sending back updated params")
                    send_packet[0] = model_version
                    send_packet[1:] = flat_params
                    # with timer.measure("push_params") as t:
                    dst = src
                    buf = send_packet.copy()
                    req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_PARAMS)
                    child_send_reqs.append((req, buf, dst))
                    t["tag"] = src

                    # -- push up local model deltas periodically --
                    if n_updates >= n_to_wait():
                        # log("received enough updates, sending up local model")
                        send_packet[1:] = anchor_params - flat_params
                        # with timer.measure("push_params") as t:
                        dst = parent
                        buf = send_packet.copy()
                        req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_GRADS)
                        parent_send_reqs.append((req, buf, dst))
                        t["tag"] = src
                        anchor_params = flat_params.copy()
                        n_updates = 0

            # -- clean up finished sends --
            for i in reversed(range(len(parent_send_reqs))):
                req, buf, dst = parent_send_reqs[i]
                if req.Test():
                    parent_send_reqs.pop(i)
            global_state["send_reqs_children"] = []
            for i in reversed(range(len(child_send_reqs))):
                req, buf, dst = child_send_reqs[i]
                if req.Test():
                    child_send_reqs.pop(i)
                else:
                    global_state["send_reqs_children"].append(dst)
                    # log(f"appended child {dst} to send reqs children")

            global_state["n_parent_send_reqs"] = len(parent_send_reqs)
            global_state["n_child_send_reqs"] = len(child_send_reqs)

    log("edge ps terminating...")

    log("cancelling parent recv req")
    cancel_recv_req(parent_recv_req)
    log("cancelling child recv reqs")
    for req in child_recv_reqs:
        cancel_recv_req(req)

    log(f"waiting to send {len(stop_reqs)} stop signals...")
    for req, _ in stop_reqs:
        safe_wait_send_req(req, timeout=2.0)

    log(f"cancelling {len(parent_send_reqs)} parent send reqs")
    for req, _, _ in parent_send_reqs:
        # safe_wait_send_req(req, timeout=0.5)
        cleanup_mpi_req(req)
        log("...")

    log(f"safe waiting {len(child_send_reqs)} child send reqs")
    for req, _, _ in child_send_reqs:
        safe_wait_send_req(req, timeout=0.5)
        log("...")

    log("edge ps done cleaning up")

    timer.print_tree()


def run_worker(model, parent, comm, n_workers, worker_idx):

    log(f"[WORKER {worker_idx}] parent: {parent}")

    global global_state
    timer = TimeTree(print_func=log, category_stack_ref=global_state["sections"])

    # -- load datasets --
    # train_ds, test_ds = load_mnist()
    train_ds, test_ds = load_cifar10()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=n_workers,
        rank=worker_idx,
        shuffle=True
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_ds,
        num_replicas=n_workers,
        rank=worker_idx,
        shuffle=True
    )

    # -- create data loaders --
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        drop_last=False,
    )

    # -- define loss function --
    loss_fn = torch.nn.CrossEntropyLoss()

    # -- initialize optimizer --
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # -- pre-allocate buffers --
    packet_size = flatten_params(model).size + 1
    recv_packet = np.empty(packet_size, dtype=np.float32)
    send_packet = np.empty(packet_size, dtype=np.float32)
    stop_buf = np.empty(1, dtype=np.int32)

    param_recv_req = comm.Irecv(recv_packet, source=parent, tag=Msg.PUSH_PARAMS)
    stop_recv_req = comm.Irecv(stop_buf, source=parent, tag=Msg.STOP)
    send_req = None

    model_version = -1
    stop_flag = False

    epoch = 0
    # for epoch in range(n_epochs):
    while True:
        # timer.reset()
        log(f"Epoch {epoch + 1}/{n_epochs}")
        with timer.measure("epoch"):

            train_sampler.set_epoch(epoch)
            model.train()

            with timer.measure("batch"):
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    with timer.measure("wait_recv"):
                        idx = MPI.Request.Waitany([param_recv_req, stop_recv_req])

                    if idx == MPI.UNDEFINED:
                        log("UNDEFINED INDEX (this should not happen)")
                        stop_flag = True
                        break

                    if idx == 1:
                        log("Stop signal received, stopping training loop")
                        stop_flag = True
                        break

                    model_version = int(recv_packet[0])
                    load_params(model, recv_packet[1:])

                    param_recv_req = comm.Irecv(recv_packet, source=parent, tag=Msg.PUSH_PARAMS)

                    with timer.measure("step"):
                        step_start = time.perf_counter()

                        # -- forward pass --
                        y_pred = model(X_batch)
                        loss = loss_fn(y_pred, y_batch)

                        predicted = torch.argmax(y_pred, 1)
                        correct = (predicted == y_batch).sum().item()
                        train_accuracy = correct/X_batch.size(0)

                        # -- backward pass --
                        model.zero_grad()
                        loss.backward()

                        # # -- update parameters --
                        # optimizer.step()
                        # optimizer.zero_grad()

                        step_time = time.perf_counter() - step_start

                    if send_req is not None:
                        with timer.measure("wait_send"):
                            send_req[0].Wait()

                    global_state["n_parent_send_reqs"] = 0

                    with timer.measure("serialize_grads"):
                        send_packet[0] = model_version
                        send_packet[1:] = flatten_grads(model)

                    # with timer.measure("push_grads"):
                    dst = parent
                    buf = send_packet.copy()
                    req = comm.Isend(buf, dest=dst, tag=Msg.PUSH_GRADS)
                    send_req = (req, buf, dst)

                    global_state["n_parent_send_reqs"] = 1

                    if (batch_idx + 1) % 50 == 0:
                        # samples_processed = batch_idx*X_batch.size(0)
                        # samples_total = len(train_loader.dataset)

                        log(
                            f"batch: {batch_idx + 1:4d}/{len(train_loader)}  {(batch_idx + 1)/len(train_loader):.0%}  \t"
                            # f"samples: {samples_processed:6d}/{samples_total} ({samples_processed/samples_total}:.0%})  \t"
                            # f"labels: {y_batch.tolist()}  \t"
                            f"step time: {step_time:.6f}s  \t"
                            f"train accuracy: {train_accuracy:.4f}  \t"
                            f"loss: {loss.item():.4f}  \t"
                            f"model version: {model_version}  \t",
                            indent=1
                        )

        timer.print_tree()

        if stop_flag:
            break

        epoch += 1

    log("worker terminating...")

    cancel_recv_req(param_recv_req)

    log("safe waiting send req")
    if send_req is not None:
        req, buf, dst = send_req
        safe_wait_send_req(req)

    log("worker done cleaning up")


def main():
    print(f"({hostname}) Rank {world_rank}/{world_size - 1}")

    # model = LeNet()
    # model = LeNetCIFAR()
    # model = ResNet20CIFAR(BasicBlock, layers=[1, 1, 1], num_classes=10)
    model = ResNet20CIFAR(BasicBlock, layers=[3, 3, 3], num_classes=10)
    # model = AlexNetCIFAR()
    model.apply(disable_state)

    # TODO: fix hierarchical being fixed and config only being hierarchical

    # topology = build_topology(world_size, hierarchical=True)
    topology = build_topology_from_yaml(config_file, world_size)

    role, parent, children, sync = get_role(world_rank, topology)

    global device_role
    device_role = role

    n_workers = len(topology.get("workers"))
    worker_comm = world.Create(world.Get_group().Incl(topology.get("workers")))

    collect_metrics = threading.Event()
    monitor_thread = threading.Thread(target=collect_sys_metrics, args=(collect_metrics,))
    monitor_thread.start()
    collect_metrics.set()

    remove_bw_cpu_limits()
    match role:
        case Role.ROOT_PS:
            run_parameter_server(model, children, world_size - 1, world, sync=sync)

        case Role.EDGE_PS:
            run_edge_parameter_server(model, parent, children, world, sync=sync)

        case Role.WORKER:
            worker_idx = worker_comm.Get_rank()
            apply_bw_cpu_limits(topology.get("bw_cpu_limits").get(world_rank))
            run_worker(model, parent, world, n_workers, worker_idx)

        case Role.EVAL:
            run_evaluator(model, 0, world)

    log("returned from role function")
    world.Barrier()
    collect_metrics.clear()
    monitor_thread.join()
    remove_bw_cpu_limits()

    # TODO: clean up the interfaces and modularize everything


if __name__ == "__main__":
    main()
