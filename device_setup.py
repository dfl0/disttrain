"""Check files, install dependencies, and prepare datasets on a device."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def check_files(root, names):
    result = {}
    for name in names:
        path = root / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    print(json.dumps(result))


def setup_environment(root, force):
    import venv

    python = root / "venv/bin/python"
    created = not python.is_file()
    if created:
        venv.create(root / "venv", with_pip=True)
    marker = root / "venv/.disttrain-requirements.sha256"
    digest = hashlib.sha256((root / "requirements_node.txt").read_bytes()).hexdigest()
    imports = "import torch, torchvision, zmq, numpy, yaml, psutil"
    healthy = subprocess.run([str(python), "-c", imports], capture_output=True).returncode == 0
    if created or force or not healthy or not marker.exists() or marker.read_text() != digest:
        command = [str(python), "-m", "pip", "install", "-r", str(root / "requirements_node.txt")]
        if force:
            command.append("--force-reinstall")
        subprocess.run(command, check=True)
        subprocess.run([str(python), "-c", imports], check=True)
        marker.write_text(digest)
        print("dependencies ready")
    else:
        print("dependencies unchanged")


def prepare_data(root, data_root, dataset):
    import os

    os.chdir(root)
    sys.path.insert(0, str(root))
    from data import load_datasets

    load_datasets(root=data_root, download=True, name=dataset)
    print("dataset ready")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    check = actions.add_parser("check")
    check.add_argument("workdir", type=Path)
    check.add_argument("files", nargs="+")
    environment = actions.add_parser("environment")
    environment.add_argument("workdir", type=Path)
    environment.add_argument("--force", action="store_true")
    dataset = actions.add_parser("dataset")
    dataset.add_argument("workdir", type=Path)
    dataset.add_argument("data_root")
    dataset.add_argument("name", choices=("mnist", "cifar10"))
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required")
    if args.action == "check":
        check_files(args.workdir, args.files)
    elif args.action == "environment":
        setup_environment(args.workdir, args.force)
    else:
        prepare_data(args.workdir, args.data_root, args.name)


if __name__ == "__main__":
    main()
