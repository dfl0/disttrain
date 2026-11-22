"""Test incremental device setup."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

import setup_devices


class SetupTests(unittest.TestCase):
    def test_remote_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.py"
            path.write_text("content")
            result = subprocess.run(
                [sys.executable, "-", "check", directory, "runtime.py", "missing.py"],
                input=Path(setup_devices.__file__).with_name("device_setup.py").read_text(),
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(json.loads(result.stdout), {
                "runtime.py": hashlib.sha256(b"content").hexdigest(), "missing.py": None,
            })

    def test_copy_only_changed_files(self):
        source = Path(setup_devices.__file__).parent
        hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                  for name in setup_devices.RUNTIME_FILES}
        with patch("setup_devices.remote") as remote, patch("setup_devices.subprocess.run") as run:
            remote.return_value.stdout = json.dumps(hashes)
            with redirect_stdout(io.StringIO()):
                setup_devices.deploy("host", "/tmp/project", "python3", setup=False, force=False, dry_run=False)
            run.assert_not_called()
            hashes["runtime.py"] = None
            remote.return_value.stdout = json.dumps(hashes)
            with redirect_stdout(io.StringIO()):
                setup_devices.deploy("host", "/tmp/project", "python3", setup=False, force=False, dry_run=False)
            copy = run.call_args_list[-1].args[0]
            self.assertEqual(copy[0], "scp")
            self.assertEqual(copy[copy.index("--") + 1:], ["runtime.py", "host:/tmp/project/"])

    def test_dry_run_deduplicates_hosts_and_selects_shared_dataset_host(self):
        source = Path(setup_devices.__file__).parent
        config = yaml.safe_load((source / "configs/example_config.yaml").read_text())
        workers = [node for node in config["nodes"] if node["role"] == "worker"]
        for node in workers:
            node["host"] = "workers"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            output = io.StringIO()
            with patch("setup_devices.subprocess.run") as run, redirect_stdout(output):
                setup_devices.main([str(path), "--workdir", "/tmp/project", "--dry-run",
                                    "--download-data", "--dataset-host", "workers"])
            run.assert_not_called()
            self.assertEqual(output.getvalue().count("Preparing workers\n"), 1)
            self.assertEqual(output.getvalue().count(": prepare cifar10 at "), 1)


if __name__ == "__main__":
    unittest.main()
