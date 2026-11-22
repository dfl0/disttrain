"""Prepare devices for a training run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from topology import load_topology


RUNTIME_FILES = (
    "dist_train.py", "launcher.py", "runtime.py", "topology.py", "lifecycle.py",
    "training.py", "ps.py", "async_ps.py", "protocol.py", "transport.py",
    "models.py", "data.py", "metrics.py", "requirements_node.txt",
)
SSH_OPTIONS = (
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
)


def remote(host, python, action, *args, capture=False):
    command = shlex.join([python, "-", action, *args])
    script = Path(__file__).with_name("device_setup.py").read_text()
    return subprocess.run(
        ["ssh", "-T", *SSH_OPTIONS, host, command],
        input=script, check=True, text=True, capture_output=capture,
    )


def deploy(host, workdir, python, *, setup, force, dry_run):
    source = Path(__file__).resolve().parent
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in RUNTIME_FILES}
    if dry_run:
        print(f"{host}: compare and copy runtime files to {workdir}", flush=True)
        if setup:
            print(f"{host}: check venv and dependencies" + ("; force reinstall" if force else ""), flush=True)
        return
    result = remote(host, python, "check", workdir, *RUNTIME_FILES, capture=True)
    existing = json.loads(result.stdout)
    changed = [name for name, digest in hashes.items() if existing.get(name) != digest]
    if changed:
        subprocess.run(
            ["ssh", "-T", *SSH_OPTIONS, host, shlex.join(["mkdir", "-p", workdir])], check=True,
        )
        subprocess.run(
            ["scp", *SSH_OPTIONS, "--", *changed, f"{host}:{workdir}/"], cwd=source, check=True,
        )
    print(f"{host}: copied {len(changed)} files" if changed else f"{host}: runtime unchanged", flush=True)
    if setup:
        remote(host, python, "environment", workdir, *(["--force"] if force else []))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="run config; prepares each distinct host")
    parser.add_argument("--host", help="prepare one host instead of a config")
    parser.add_argument("--workdir", required=True, help="absolute remote project directory")
    parser.add_argument("--python", default="python3", help="remote Python for checks and venv creation")
    parser.add_argument("--setup", action="store_true", help="also prepare dependencies in single-host mode")
    parser.add_argument("--force", action="store_true", help="reinstall dependencies even if checks pass")
    parser.add_argument("--dry-run", action="store_true", help="show planned actions without contacting hosts")
    parser.add_argument("--download-data", action="store_true", help="prepare both dataset splits on worker/evaluator hosts")
    parser.add_argument("--dataset-host", action="append", default=[],
                        help="with --download-data, prepare only this host for shared storage; repeat per storage location")
    args = parser.parse_args(argv)
    if bool(args.config) == bool(args.host):
        parser.error("provide either a config or --host")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.workdir) or Path(args.workdir) == Path("/"):
        parser.error("--workdir requires an absolute project path without spaces or shell characters")
    if not re.fullmatch(r"[A-Za-z0-9_./][A-Za-z0-9_./-]*", args.python):
        parser.error("--python requires an executable name or path without spaces or shell characters")
    if args.host and not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*", args.host):
        parser.error("invalid SSH host")
    if args.dataset_host and not args.download_data:
        parser.error("--dataset-host requires --download-data")
    if args.download_data and not args.config:
        parser.error("--download-data requires a config")
    setup = bool(args.config) or args.setup
    if args.force and not setup:
        parser.error("--force requires a config or --setup")
    try:
        topology = load_topology(args.config) if args.config else None
        hosts = list(dict.fromkeys(node["host"] for node in topology["node_configs"].values())) if topology else [args.host]
        data_hosts = []
        if args.download_data:
            data_hosts = list(dict.fromkeys(args.dataset_host or [
                node["host"] for node in topology["node_configs"].values() if node["role"] in ("worker", "eval")
            ]))
            if not set(data_hosts) <= set(hosts):
                parser.error("--dataset-host must be a host in the config")
        for host in hosts:
            print(f"Preparing {host}", flush=True)
            deploy(host, args.workdir, args.python, setup=setup, force=args.force, dry_run=args.dry_run)
        for host in data_hosts:
            dataset = "mnist" if topology["workload"] == "lenet_mnist" else "cifar10"
            data_root = topology["data"]["root"]
            if args.dry_run:
                print(f"{host}: prepare {dataset} at {data_root} (relative paths use {args.workdir})", flush=True)
            else:
                remote(host, f"{args.workdir}/venv/bin/python", "dataset", args.workdir, data_root, dataset)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        parser.exit(1, f"Setup failed: {detail}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Setup failed: {error}\n")


if __name__ == "__main__":
    main()
