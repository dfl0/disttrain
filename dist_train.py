import time
import os
import shutil
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

from timetree import TimeTree
from comm import Role, Msg, cancel_recv_req, safe_wait_send_req, cleanup_mpi_req
from ps import ParameterServer


warnings.filterwarnings("ignore", module="torchvision.datasets.cifar")


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

    eval_cfg = root.get("eval", {})
    eval_accuracy_threshold = eval_cfg.get("accuracy_threshold", 0.90)
    eval_time_limit_s = eval_cfg.get("time_limit_s", 3600)

    training_cfg = root.get("training", {})
    training = {
        "learning_rate": training_cfg.get("learning_rate", 0.01),
        "batch_size":    training_cfg.get("batch_size", 32),
        "n_epochs":      training_cfg.get("n_epochs", 1),
        "eval_interval": training_cfg.get("eval_interval", 200),
        "local_steps":   training_cfg.get("local_steps", 1),
    }

    return {
        "central_server": central_server,
        "edge_servers": edge_servers,
        "workers": workers,
        "parents": parents,
        "children": children,
        "sync": sync,
        "bw_cpu_limits": specs,
        "eval_accuracy_threshold": eval_accuracy_threshold,
        "eval_time_limit_s": eval_time_limit_s,
        "training": training,
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

    with open("/sys/fs/cgroup/train_cpu/cgroup.procs", "w") as f:
        f.write("0")
    run_cmd("rmdir /sys/fs/cgroup/train_cpu")


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


class ParamSerializer:
    """
    Caches the model's parameter layout so that flatten/load operations
    do not re-walk model.parameters() on every call.
    """

    def __init__(self, model):
        layout = []
        offset = 0
        for p in model.parameters():
            n = p.numel()
            layout.append((p, offset, n))
            offset += n
        self._layout = layout
        self.n_params = offset

    def flatten_params_into(self, out):
        for p, off, n in self._layout:
            src = p.data.view(-1)
            if src.dtype != torch.float32:
                src = src.float()
            out[off:off + n] = src.detach().numpy()

    def flatten_grads_into(self, out):
        for p, off, n in self._layout:
            if p.grad is None:
                out[off:off + n] = 0.0
            else:
                src = p.grad.view(-1).detach()
                if src.dtype != torch.float32:
                    src = src.float()
                out[off:off + n] = src.numpy()

    def load_params(self, flat):
        flat_t = torch.from_numpy(flat)
        for p, off, n in self._layout:
            p.data.copy_(flat_t[off:off + n].view_as(p))


def init_params(m):
    if type(m) is torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def disable_state(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.track_running_stats = False
        m.running_mean = None
        m.running_var = None


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


def collect_sys_metrics(collect_metrics, interval=0.04, quiet=False):
    collect_metrics.wait()

    def get_net_bytes():
        counter = psutil.net_io_counters(pernic=True)["eth0"]
        return (counter.bytes_sent, counter.bytes_recv)

    t0 = time.perf_counter()
    tot_bytes = get_net_bytes()
    psutil.cpu_percent()  # prime the baseline; first call always returns 0.0
    while collect_metrics.is_set():
        last_tot_bytes = tot_bytes

        # cpu = psutil.cpu_percent(min(interval, 0.1))
        cpu = psutil.cpu_percent(interval=min(interval, 0.1))
        freq_info = psutil.cpu_freq()
        freq = freq_info.current if freq_info else float("nan")
        mem = psutil.virtual_memory().percent
        temp_entries = psutil.sensors_temperatures().get("cpu_thermal", [None])
        temp = temp_entries[0].current if temp_entries[0] else float("nan")

        if shutil.which("vcgencmd") is not None:
            throttled_raw = run_cmd("vcgencmd get_throttled")
            throttled_val = int(throttled_raw.split('=')[-1], 16)
            throttled_now = (throttled_val >> 2) % 2   # if the pi is currently throttled
            throttled_bef = (throttled_val >> 18) % 2  # if throttling has occured before
        else:
            throttled_now = throttled_bef = float("nan")

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

def run_evaluator(model, root, comm, accuracy_threshold=0.90, time_limit_s=3600, batch_size=32):
    log("[EVALUATOR]")

    # -- load test data  --
    # _, test_ds = load_mnist()
    _, test_ds = load_cifar10()

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        drop_last=False,
        num_workers=2,
        pin_memory=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    global global_state
    timer = TimeTree(print_func=log)

    serializer = ParamSerializer(model)
    flat_params = np.empty(serializer.n_params, dtype=np.float32)
    packet_size = serializer.n_params + 1

    test_acc = 0
    model_version = 0

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

            if test_acc >= accuracy_threshold or done_time >= time_limit_s:
            # if True:
                log("sending STOP message...")
                # drain any remaining eval pushes so root does not hold dangling sends
                while comm.Iprobe(source=0, tag=Msg.EVAL):
                    comm.Recv(np.empty(packet_size, dtype=np.float32), source=0, tag=Msg.EVAL)
                comm.Send(np.array([0], dtype=np.int32), dest=0, tag=Msg.STOP)
                break

        else:
            # log(f"No updated parameters received")
            time.sleep(0.5)

    log("evaluator terminating...")
    log("evaluator done")


def run_worker(model, parent, comm, n_workers, worker_idx, batch_size=32, n_epochs=1, local_steps=1):

    log(f"[WORKER {worker_idx}] parent: {parent}, local_steps: {local_steps}")

    global global_state
    timer = TimeTree(print_func=log)

    # -- load datasets --
    # train_ds, test_ds = load_mnist()
    train_ds, test_ds = load_cifar10()

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
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
        num_workers=2,
        pin_memory=False,
    )

    # -- define loss function --
    loss_fn = torch.nn.CrossEntropyLoss()

    # -- pre-allocate buffers --
    serializer  = ParamSerializer(model)
    packet_size = serializer.n_params + 1
    recv_packet = np.empty(packet_size, dtype=np.float32)
    grad_buf    = np.empty(packet_size, dtype=np.float32)  # owned send buffer, reused each step
    stop_buf    = np.empty(1, dtype=np.int32)

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
                local_step = 0        # counts micro-steps within the current local window
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

                    # fetch fresh params from PS at the start of each local window
                    if local_step == 0:
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
                        serializer.load_params(recv_packet[1:])

                        param_recv_req = comm.Irecv(recv_packet, source=parent, tag=Msg.PUSH_PARAMS)

                        model.zero_grad()

                    with timer.measure("step"):
                        step_start = time.perf_counter()

                        y_pred = model(X_batch)
                        loss = loss_fn(y_pred, y_batch)

                        predicted = torch.argmax(y_pred, 1)
                        correct = (predicted == y_batch).sum().item()
                        train_accuracy = correct/X_batch.size(0)

                        # accumulate gradients across local steps
                        loss.backward()

                        step_time = time.perf_counter() - step_start

                    local_step += 1

                    # push accumulated gradients after local_steps micro-steps
                    if local_step == local_steps:
                        if send_req is not None:
                            with timer.measure("wait_send"):
                                send_req[0].Wait()

                        global_state["n_parent_send_reqs"] = 0

                        with timer.measure("serialize_grads"):
                            grad_buf[0] = model_version
                            serializer.flatten_grads_into(grad_buf[1:])

                        req = comm.Isend(grad_buf, dest=parent, tag=Msg.PUSH_GRADS)
                        send_req = (req, grad_buf, parent)

                        global_state["n_parent_send_reqs"] = 1
                        local_step = 0

                    if (batch_idx + 1) % 50 == 0:
                        log(
                            f"batch: {batch_idx + 1:4d}/{len(train_loader)}  {(batch_idx + 1)/len(train_loader):.0%}  \t"
                            f"step time: {step_time:.6f}s  \t"
                            f"train accuracy: {train_accuracy:.4f}  \t"
                            f"loss: {loss.item():.4f}  \t"
                            f"model version: {model_version}  \t",
                            indent=1
                        )

                # flush any partially-accumulated gradients at epoch end
                if not stop_flag and local_step > 0:
                    if send_req is not None:
                        send_req[0].Wait()
                    grad_buf[0] = model_version
                    serializer.flatten_grads_into(grad_buf[1:])
                    req = comm.Isend(grad_buf, dest=parent, tag=Msg.PUSH_GRADS)
                    send_req = (req, grad_buf, parent)

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

    topology  = build_topology_from_yaml(config_file, world_size)
    training  = topology["training"]
    serializer = ParamSerializer(model)

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
            ParameterServer(
                model=model, children=children, comm=world, sync=sync,
                serializer=serializer, init_params=init_params,
                parent=None, evaluator=world_size - 1,
                lr=training["learning_rate"],
                eval_interval=training["eval_interval"],
                global_state=global_state, log=log,
            ).run()

        case Role.EDGE_PS:
            ParameterServer(
                model=model, children=children, comm=world, sync=sync,
                serializer=serializer, init_params=init_params,
                parent=parent,
                lr=training["learning_rate"],
                eval_interval=training["eval_interval"],
                global_state=global_state, log=log,
            ).run()

        case Role.WORKER:
            worker_idx = worker_comm.Get_rank()
            apply_bw_cpu_limits(topology.get("bw_cpu_limits").get(world_rank))
            run_worker(model, parent, world, n_workers, worker_idx,
                       batch_size=training["batch_size"],
                       n_epochs=training["n_epochs"],
                       local_steps=training["local_steps"])

        case Role.EVAL:
            run_evaluator(model, 0, world,
                          accuracy_threshold=topology.get("eval_accuracy_threshold"),
                          time_limit_s=topology.get("eval_time_limit_s"),
                          batch_size=training["batch_size"])

    log("returned from role function")
    world.Barrier()
    collect_metrics.clear()
    monitor_thread.join()
    remove_bw_cpu_limits()


if __name__ == "__main__":
    main()
