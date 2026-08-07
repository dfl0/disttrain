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
import zmq

from timetree import TimeTree
from comm import Role, Msg
from ps import ParameterServer


warnings.filterwarnings("ignore", module="torchvision.datasets.cifar")


# ----  get node/proc info  --------------------------------------------------+

hostname = socket.gethostname()
config_file = 'config.yaml'

device_role    = None
global_start_time = None  # set in main() after topology is parsed

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

def build_topology_from_yaml(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    base_port   = raw.get("base_port", 5550)
    bind_addr   = raw.get("bind_addr", "127.0.0.1")
    root_addr   = bind_addr

    eval_cfg = raw.get("eval", {})
    eval_host = eval_cfg.get("host", "127.0.0.1")
    eval_port = eval_cfg.get("port", base_port + 10)
    eval_accuracy_threshold = eval_cfg.get("accuracy_threshold", 0.90)
    eval_time_limit_s       = eval_cfg.get("time_limit_s", 3600)

    training_cfg = raw.get("training", {})
    training = {
        "learning_rate": training_cfg.get("learning_rate", 0.01),
        "batch_size":    training_cfg.get("batch_size", 32),
        "n_epochs":      training_cfg.get("n_epochs", 1),
        "eval_interval": training_cfg.get("eval_interval", 200),
        "local_steps":   training_cfg.get("local_steps", 1),
    }

    # build role-specific config keyed by hostname/address
    node_configs = {}

    # root PS
    node_configs[_addr_key(root_addr)] = {
        "role": Role.ROOT_PS,
        "sync": raw.get("sync", False),
        "bind_addr": root_addr,
        "base_port": base_port,
        "children_hostnames": [],  # filled below
    }

    all_worker_specs = {}  # hostname → {bw, cpu, parent_addr}

    if "edge_servers" in raw:
        for edge in raw["edge_servers"]:
            edge_addr = edge["bind_addr"]
            child_hostnames = [w["hostname"] for w in edge.get("workers", [])]
            node_configs[_addr_key(edge_addr)] = {
                "role": Role.EDGE_PS,
                "sync": edge.get("sync", False),
                "bind_addr": edge_addr,
                "base_port": base_port,
                "parent_addr": root_addr,
                "children_hostnames": child_hostnames,
            }
            node_configs[_addr_key(root_addr)]["children_hostnames"].append(edge_addr)
            for w in edge.get("workers", []):
                all_worker_specs[w["hostname"]] = {
                    "bw": w.get("bw"), "cpu": w.get("cpu"),
                    "parent_addr": edge_addr,
                }
    elif "workers" in raw:
        child_hostnames = [w["hostname"] for w in raw["workers"]]
        node_configs[_addr_key(root_addr)]["children_hostnames"] = child_hostnames
        for w in raw["workers"]:
            all_worker_specs[w["hostname"]] = {
                "bw": w.get("bw"), "cpu": w.get("cpu"),
                "parent_addr": root_addr,
            }

    for whost, spec in all_worker_specs.items():
        node_configs[whost] = {"role": Role.WORKER, **spec}

    node_configs[_addr_key(eval_host)] = {
        "role": Role.EVAL,
        "eval_host": eval_host,
        "eval_port": eval_port,
        "root_addr": root_addr,
        "base_port": base_port,
    }

    return {
        "node_configs":             node_configs,
        "root_addr":                root_addr,
        "base_port":                base_port,
        "eval_host":                eval_host,
        "eval_port":                eval_port,
        "eval_accuracy_threshold":  eval_accuracy_threshold,
        "eval_time_limit_s":        eval_time_limit_s,
        "training":                 training,
        "worker_specs":             all_worker_specs,
    }


def _addr_key(addr):
    """Normalise address to a bare hostname/IP for dict key lookup."""
    return addr.replace("tcp://", "").split(":")[0]


def determine_role(this_hostname, topology):
    """Match this process's hostname against the topology and return its config."""
    key = _addr_key(this_hostname)
    cfg = topology["node_configs"].get(key)
    if cfg is None:
        # also check bare hostname match (worker hostnames are not IP addresses)
        cfg = topology["node_configs"].get(this_hostname)
    if cfg is None:
        raise RuntimeError(
            f"Hostname {this_hostname!r} not found in config. "
            f"Known keys: {list(topology['node_configs'].keys())}"
        )
    return cfg


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
    if bw_cpu_limits[0]:
        run_cmd("tc qdisc add dev eth0 root handle 1: htb default 1")
        run_cmd(f"tc class add dev eth0 parent 1: classid 1:1 htb rate {bw_cpu_limits[0]}")
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
    run_cmd("tc qdisc del root dev eth0")
    run_cmd("tc qdisc del dev eth0 handle ffff: ingress")
    run_cmd("tc qdisc del root dev ifb0")
    run_cmd("ip link set dev ifb0 down")
    run_cmd("modprobe -r ifb")

    with open("/sys/fs/cgroup/train_cpu/cgroup.procs", "w") as f:
        f.write("0")
    run_cmd("rmdir /sys/fs/cgroup/train_cpu")


def log(msg, indent=0):
    t = time.perf_counter() - global_start_time
    whole = int(t)
    ms = int((t - whole) * 1000)
    timestamp = time.strftime("%H:%M:%S", time.gmtime(whole))
    timestamp = f"{timestamp}:{ms:03d}"

    global device_role
    assert device_role

    # use last segment of hostname (e.g. "pi-worker-1" → "worker-1") for compact prefix
    short_host = hostname.split("-", 1)[-1] if "-" in hostname else hostname
    prefix = (
        f"({timestamp}) "
        f"{short_host:<12} "
        f"[{device_role.name:<{8}}]  "
    )

    indent_str = "    " * indent

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
            # stagger log output across processes using a hostname-derived offset
            host_offset = (hash(hostname) & 0xFF) / 255.0
            time.sleep(host_offset * interval)
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

def run_evaluator(pull_sock, push_sock, model, accuracy_threshold=0.90, time_limit_s=3600, batch_size=32):
    log("[EVALUATOR]")

    # -- load test data  --
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

    serializer  = ParamSerializer(model)
    flat_params = np.empty(serializer.n_params, dtype=np.float32)

    test_acc     = 0
    model_version = 0
    new_params   = False

    start_time = time.perf_counter()
    while True:
        # -- drain all parameter pushes from root to get newest version --
        with timer.measure("drain"):
            drained = 0
            while pull_sock.poll(timeout=0):
                frames = pull_sock.recv_multipart()
                # frames: [msg_type_byte, payload]
                payload     = frames[1] if len(frames) > 1 else b''
                packet      = np.frombuffer(payload, dtype=np.float32)
                new_version = int(packet[0])
                drained += 1

                if new_version > model_version:
                    model_version = new_version
                    flat_params[:] = packet[1:]
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

                        total_loss += loss.item() * test_batch_size
                        total_samples += test_batch_size

                test_acc  = total_correct / total_samples
                test_loss = total_loss / total_samples

            done_time = time.perf_counter() - start_time
            log('*' * 100)
            log(f"Evaluation (recv'd after {recv_time:.2f}s, done after {done_time:.2f}s): version {model_version}, accuracy: {test_acc:.2%}, loss: {test_loss:.4f}")
            log('*' * 100)

            if test_acc >= accuracy_threshold or done_time >= time_limit_s:
                log("sending STOP message...")
                # drain any remaining eval pushes before signalling stop
                while pull_sock.poll(timeout=0):
                    pull_sock.recv_multipart()
                push_sock.send_multipart([bytes([Msg.STOP]), b''])
                break

        else:
            pull_sock.poll(timeout=500)   # wait up to 500 ms for new params

    log("evaluator terminating...")

    pull_sock.setsockopt(zmq.LINGER, 0)
    pull_sock.close()
    push_sock.setsockopt(zmq.LINGER, 0)
    push_sock.close()

    log("evaluator done")


def run_worker(dealer_sock, model, n_workers, batch_size=32, n_epochs=1, local_steps=1):

    # -- registration: send hostname, get assigned worker_idx --
    dealer_sock.send_multipart([bytes([Msg.REGISTER]), hostname.encode()])
    frames    = dealer_sock.recv_multipart()
    assert frames[0] == bytes([Msg.ASSIGN])
    worker_idx = int(np.frombuffer(frames[1], dtype=np.int32)[0])

    # -- receive initial params --
    frames = dealer_sock.recv_multipart()
    assert frames[0] == bytes([Msg.PUSH_PARAMS])
    init_packet   = np.frombuffer(frames[1], dtype=np.float32)
    model_version = int(init_packet[0])

    log(f"[WORKER {worker_idx}] registered, initial version {model_version}, local_steps: {local_steps}")

    serializer = ParamSerializer(model)
    serializer.load_params(init_packet[1:])

    global global_state
    timer = TimeTree(print_func=log)

    # -- load datasets --
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

    # -- pre-allocate send buffer --
    packet_size = serializer.n_params + 1
    grad_buf    = np.empty(packet_size, dtype=np.float32)

    stop_flag = False
    epoch = 0

    while True:
        log(f"Epoch {epoch + 1}/{n_epochs}")
        with timer.measure("epoch"):

            train_sampler.set_epoch(epoch)
            model.train()

            with timer.measure("batch"):
                local_step = 0
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

                    # fetch fresh params from PS at start of each local window
                    if local_step == 0:
                        with timer.measure("wait_recv"):
                            # poll blocks until a message arrives
                            dealer_sock.poll()
                            frames    = dealer_sock.recv_multipart()
                            msg_type  = frames[0]
                            payload   = frames[1] if len(frames) > 1 else b''

                        if msg_type == bytes([Msg.STOP]):
                            log("Stop signal received, stopping training loop")
                            stop_flag = True
                            break

                        assert msg_type == bytes([Msg.PUSH_PARAMS])
                        packet = np.frombuffer(payload, dtype=np.float32)
                        model_version = int(packet[0])
                        serializer.load_params(packet[1:])
                        model.zero_grad()

                    with timer.measure("step"):
                        step_start = time.perf_counter()

                        y_pred = model(X_batch)
                        loss = loss_fn(y_pred, y_batch)

                        predicted = torch.argmax(y_pred, 1)
                        correct = (predicted == y_batch).sum().item()
                        train_accuracy = correct / X_batch.size(0)

                        # accumulate gradients across local steps
                        loss.backward()

                        step_time = time.perf_counter() - step_start

                    local_step += 1

                    # push accumulated gradients after local_steps micro-steps
                    if local_step == local_steps:
                        with timer.measure("serialize_grads"):
                            grad_buf[0] = model_version
                            serializer.flatten_grads_into(grad_buf[1:])

                        dealer_sock.send_multipart([bytes([Msg.PUSH_GRADS]), grad_buf.tobytes()])
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
                    grad_buf[0] = model_version
                    serializer.flatten_grads_into(grad_buf[1:])
                    dealer_sock.send_multipart([bytes([Msg.PUSH_GRADS]), grad_buf.tobytes()])

        timer.print_tree()

        if stop_flag:
            break

        epoch += 1

    log("worker terminating...")
    dealer_sock.setsockopt(zmq.LINGER, 0)
    dealer_sock.close()
    log("worker done cleaning up")


def main():
    global device_role, global_start_time

    topology  = build_topology_from_yaml(config_file)
    role_cfg  = determine_role(hostname, topology)
    role      = role_cfg["role"]
    training  = topology["training"]

    device_role       = role
    global_start_time = time.perf_counter()

    log(f"({hostname}) starting as {role.name}")

    # model = LeNet()
    # model = LeNetCIFAR()
    # model = ResNet20CIFAR(BasicBlock, layers=[1, 1, 1], num_classes=10)
    model = ResNet20CIFAR(BasicBlock, layers=[3, 3, 3], num_classes=10)
    # model = AlexNetCIFAR()
    model.apply(disable_state)

    serializer = ParamSerializer(model)
    ctx = zmq.Context()

    collect_metrics = threading.Event()
    monitor_thread  = threading.Thread(target=collect_sys_metrics, args=(collect_metrics,))
    monitor_thread.start()
    collect_metrics.set()

    remove_bw_cpu_limits()

    base_port = topology["base_port"]
    eval_host = topology["eval_host"]
    eval_port = topology["eval_port"]
    root_addr = topology["root_addr"]

    match role:
        case Role.ROOT_PS:
            router = ctx.socket(zmq.ROUTER)
            router.bind(f"tcp://*:{base_port}")
            stop_pull = ctx.socket(zmq.PULL)
            stop_pull.bind(f"tcp://*:{base_port + 1}")
            eval_push = ctx.socket(zmq.PUSH)
            eval_push.connect(f"tcp://{eval_host}:{eval_port}")
            ParameterServer(
                model=model,
                children_hostnames=role_cfg["children_hostnames"],
                sync=role_cfg["sync"],
                router_sock=router,
                stop_sock=stop_pull,
                eval_push_sock=eval_push,
                serializer=serializer,
                init_params=init_params,
                parent=None,
                lr=training["learning_rate"],
                eval_interval=training["eval_interval"],
                global_state=global_state,
                log=log,
            ).run()

        case Role.EDGE_PS:
            router = ctx.socket(zmq.ROUTER)
            router.bind(f"tcp://*:{base_port}")
            dealer = ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.IDENTITY, hostname.encode())
            dealer.connect(f"tcp://{role_cfg['parent_addr']}:{base_port}")
            ParameterServer(
                model=model,
                children_hostnames=role_cfg["children_hostnames"],
                sync=role_cfg["sync"],
                router_sock=router,
                dealer_sock=dealer,
                serializer=serializer,
                init_params=init_params,
                parent=role_cfg["parent_addr"],
                lr=training["learning_rate"],
                eval_interval=training["eval_interval"],
                global_state=global_state,
                log=log,
            ).run()

        case Role.WORKER:
            spec = role_cfg
            apply_bw_cpu_limits([spec.get("bw"), spec.get("cpu")])
            all_workers = [
                w for edge in topology.get("edge_servers_raw", [])
                for w in edge.get("workers", [])
            ] or list(topology.get("worker_specs", {}).keys())
            n_workers = len(topology["worker_specs"])
            dealer = ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.IDENTITY, hostname.encode())
            dealer.connect(f"tcp://{spec['parent_addr']}:{base_port}")
            run_worker(
                dealer_sock=dealer,
                model=model,
                n_workers=n_workers,
                batch_size=training["batch_size"],
                n_epochs=training["n_epochs"],
                local_steps=training["local_steps"],
            )

        case Role.EVAL:
            pull = ctx.socket(zmq.PULL)
            pull.bind(f"tcp://*:{eval_port}")
            push = ctx.socket(zmq.PUSH)
            push.connect(f"tcp://{root_addr}:{base_port + 1}")
            run_evaluator(
                pull_sock=pull,
                push_sock=push,
                model=model,
                accuracy_threshold=topology["eval_accuracy_threshold"],
                time_limit_s=topology["eval_time_limit_s"],
                batch_size=training["batch_size"],
            )

    log("returned from role function")
    collect_metrics.clear()
    monitor_thread.join()
    remove_bw_cpu_limits()
    ctx.term()


if __name__ == "__main__":
    main()
