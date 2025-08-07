#!/usr/bin/env python
"""
AI Logging Test Suite for dftracer
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from time import sleep
from typing import Optional

import numpy as np
import pytest


def get_dftracer_preload_path():
    """Find the path to libdftracer_preload.so in the virtual environment"""
    try:
        import site

        # Get all site-packages directories
        site_packages_dirs = site.getsitepackages()

        # Also check the user site-packages if it exists
        try:
            site_packages_dirs.append(site.getusersitepackages())
        except AttributeError:
            pass

        # Check for dftracer_libs in each site-packages directory
        for site_dir in site_packages_dirs:
            dftracer_libs_path = os.path.join(site_dir, "dftracer_libs")

            if os.path.isdir(dftracer_libs_path):
                # Try both possible locations
                lib64_path = os.path.join(
                    dftracer_libs_path, "lib64", "libdftracer_preload.so"
                )
                lib_path = os.path.join(
                    dftracer_libs_path, "lib", "libdftracer_preload.so"
                )

                if os.path.exists(lib64_path):
                    return lib64_path
                elif os.path.exists(lib_path):
                    return lib_path

        print("Warning: dftracer_libs directory or libdftracer_preload.so not found")
        return ""

    except Exception as e:
        print(f"Warning: Error finding dftracer_libs: {e}")
        return ""


@dataclass
class Args:
    log_dir: str
    data_dir: str
    disable_ai_cat: Optional[str] = None
    num_files: Optional[int] = None
    niter: Optional[int] = None
    epoch_as_metadata: bool = False
    record_size: int = 1048576


def run_single_ai_logging_test(test_config):
    """Run a single AI logging test in isolation - suitable for subprocess execution"""
    # Create test directories with unique names per test
    base_dir = os.path.join(os.path.dirname(__file__), "test_ai_logging_subprocess")
    test_name = f"{test_config['name']}_niter{test_config['niter']}_files{test_config['num_files']}"
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    log_dir = os.path.join(test_base_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set environment variables
    for env_var, value in test_config["env"].items():
        os.environ[env_var] = value

    # Set DFTRACER_DATA_DIR if specified
    if "DFTRACER_DATA_DIR" in test_config["env"]:
        os.environ["DFTRACER_DATA_DIR"] = test_config["env"]["DFTRACER_DATA_DIR"]
    else:
        os.environ["DFTRACER_DATA_DIR"] = data_dir

    # Configure log file
    log_file = os.path.join(log_dir, f"{test_config['name']}.pfw")
    if "DFTRACER_LOG_FILE" in test_config["env"]:
        log_file = test_config["env"]["DFTRACER_LOG_FILE"].replace(
            "${test_name}", test_config["name"]
        )
    os.environ["DFTRACER_LOG_FILE"] = log_file

    from dftracer.logger import ai, dftracer

    class IOHandler:
        @ai.data.item
        def read(self, filename: str):
            return np.load(filename)

        def write(self, filename: str, a):
            with open(filename, "wb") as f:
                np.save(f, a)

    def data_gen(args: Args, io: IOHandler, data: np.ndarray):
        for i in range(args.num_files):
            io.write(f"{args.data_dir}/npz/{i}-of-{args.num_files}.npy", data)

    @ai.dataloader.fetch
    def read_data(args: Args, io: IOHandler, epoch: int):
        for i in range(args.num_files):
            yield io.read(f"{args.data_dir}/npz/{i}-of-{args.num_files}.npy")

    @ai.data.preprocess.derive(name="collate")
    def collate(data):
        return data

    @ai.device.transfer
    def transfer(data):
        sleep(0.1)
        return data

    @ai.compute.forward
    def forward(data):
        sleep(0.1)
        return 0.0

    @ai.compute.backward
    def backward():
        sleep(0.1)
        with ai.comm.all_reduce(enable=False):
            sleep(0.1)

    @ai.compute
    def compute(data):
        _ = forward(data)
        backward()
        return _

    class Checkpoint:
        @ai.checkpoint.init
        def __init__(self):
            sleep(0.1)

        @ai.checkpoint.capture
        def capture(self, _):
            sleep(0.1)
            return _

        @ai.checkpoint.restart
        def restart(self, _):
            sleep(0.1)
            return _

    class Hook:
        def before_step(self, *args, **kwargs):
            ai.compute.step.start()

        def after_step(self, *args, **kwargs):
            ai.compute.step.stop()

    @ai.pipeline.train
    def train(args: Args, hook: Hook):
        io = IOHandler()

        os.makedirs(f"{args.log_dir}/npz", exist_ok=True)
        os.makedirs(f"{args.data_dir}/npz", exist_ok=True)
        data = np.ones((args.record_size, 1), dtype=np.uint8)
        data_gen(args, io, data)

        checkpoint = Checkpoint()

        checkpoint.restart({})

        if args.epoch_as_metadata:
            for epoch in range(args.niter):
                ai.pipeline.epoch.start(metadata=True)
                for step, data in ai.dataloader.fetch.iter(
                    enumerate(read_data(args, io, epoch))
                ):
                    hook.before_step()
                    data = collate(data)
                    _ = transfer(data)
                    _ = compute(data)
                    hook.after_step()
                    ai.update(step=step, epoch=epoch)
                ai.pipeline.epoch.stop(metadata=True)
        else:
            for epoch in ai.pipeline.epoch.iter(range(args.niter)):
                for step, data in ai.dataloader.fetch.iter(
                    enumerate(read_data(args, io, epoch))
                ):
                    hook.before_step()
                    data = collate(data)
                    _ = transfer(data)
                    _ = compute(data)
                    hook.after_step()
                    ai.update(step=step, epoch=epoch)

        checkpoint.capture({})

    # Apply AI category disabling
    if test_config.get("disable_ai_cat") == "all":
        ai.disable()
    elif test_config.get("disable_ai_cat") == "dataloader":
        ai.dataloader.disable()
    elif test_config.get("disable_ai_cat") == "device":
        ai.device.disable()
    elif test_config.get("disable_ai_cat") == "compute":
        ai.compute.disable()
    elif test_config.get("disable_ai_cat") == "comm":
        ai.comm.disable()
    elif test_config.get("disable_ai_cat") == "ckpt":
        ai.checkpoint.disable()

    args = Args(
        log_dir=log_dir,
        data_dir=data_dir,
        disable_ai_cat=test_config.get("disable_ai_cat"),
        num_files=test_config["num_files"],
        niter=test_config["niter"],
        epoch_as_metadata=test_config.get("epoch_as_metadata", False),
        record_size=test_config.get("record_size", 1048576),
    )

    print(
        f"Running AI logging test {test_config['name']} with log file: {log_file}, args = {args}"
    )

    try:
        hook = Hook()
        df_logger = dftracer.initialize_log(logfile=None, data_dir=None, process_id=-1)
        train(args, hook)
        df_logger.finalize()

        # Count events in log file (if dftracer is enabled)
        event_count = 0
        if test_config["env"].get("DFTRACER_ENABLE") != "0":
            # Find log files matching pattern
            log_pattern = log_file.replace(".pfw", "*-app.pfw")
            log_files = glob.glob(log_pattern)

            # Also check the exact log file name
            if not log_files and os.path.exists(log_file):
                log_files = [log_file]

            # Also check for files without the -app suffix
            if not log_files:
                log_pattern_simple = log_file.replace(".pfw", "*.pfw")
                log_files = glob.glob(log_pattern_simple)

            print(f"Looking for log files with pattern: {log_pattern}")
            print(f"Found log files: {log_files}")

            for log_file_path in log_files:
                if os.path.exists(log_file_path):
                    try:
                        with open(log_file_path) as f:
                            lines = f.readlines()
                            event_count += len(lines)
                        print(f"Found {len(lines)} events in {log_file_path}")
                    except Exception as e:
                        print(f"Error reading {log_file_path}: {e}")

            # If still no events found, check if any files were created in the log directory
            if event_count == 0:
                log_dir_files = os.listdir(os.path.dirname(log_file))
                print(f"Files in log directory: {log_dir_files}")
                # Check if pydftracer is actually available
                try:
                    import pydftracer  # noqa: F401

                    print("pydftracer library is available")
                except ImportError:
                    print(
                        "pydftracer library is NOT available - tests will run in no-op mode"
                    )
                    # Adjust expected counts for no-op mode
                    if "expected_events" in test_config:
                        test_config["expected_events"] = 0

        print(f"Test {test_config['name']} completed with {event_count} events")

        # Verify expected event count
        expected_count = test_config.get("expected_events", 0)
        if test_config["env"].get("DFTRACER_ENABLE") == "0":
            # When dftracer is disabled, expect no events
            assert event_count == 0, (
                f"Expected 0 events when DFTRACER_ENABLE=0 but got {event_count} for test {test_config['name']}"
            )
        elif expected_count > 0:
            # When a specific count is expected, check it
            assert event_count == expected_count, (
                f"Expected {expected_count} events but got {event_count} for test {test_config['name']}"
            )
        else:
            # Normal case - should have some events
            assert event_count > 5, (
                f"Expected some events but got {event_count} for test {test_config['name']}"
            )
    finally:
        shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


class TestAILogging:
    @pytest.mark.subprocess
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "normal",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "num_files": 2,
                "niter": 3,
                "expected_events": 75,
            },
            {
                "name": "epoch_as_metadata",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "num_files": 2,
                "niter": 3,
                "epoch_as_metadata": True,
                "expected_events": 75,
            },
            {
                "name": "disable_only",
                "env": {
                    "DFTRACER_ENABLE": "0",
                },
                "num_files": 2,
                "niter": 3,
                "expected_events": 0,
            },
            {
                "name": "disable_cat_all",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "all",
                "expected_events": 8,
            },
            {
                "name": "disable_cat_dataloader",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "dataloader",
                "expected_events": 60,
            },
            {
                "name": "disable_cat_device",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "device",
                "expected_events": 69,
            },
            {
                "name": "disable_cat_compute",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "compute",
                "expected_events": 51,
            },
            {
                "name": "disable_cat_ckpt",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "ckpt",
                "expected_events": 72,
            },
        ],
    )
    def test_ai_logging_subprocess_execution(self, test_config):
        """Run each AI logging test configuration in a separate subprocess"""

        # Create a temporary Python script that runs the test
        script_content = f'''
import sys
import os
sys.path.insert(0, "{os.path.dirname(os.path.dirname(__file__))}")

from tests.test_ai_logging import run_single_ai_logging_test

test_config = {test_config!r}

if __name__ == "__main__":
    try:
        result = run_single_ai_logging_test(test_config)
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test failed with error: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Run the test in a subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=120,  # Longer timeout for AI logging tests
            )

            print(f"Test {test_config['name']} subprocess output:")
            print(result.stdout)
            if result.stderr:
                print(f"Test {test_config['name']} subprocess errors:")
                print(result.stderr)

            assert result.returncode == 0, (
                f"Test {test_config['name']} failed in subprocess"
            )

        finally:
            # Clean up temporary script
            if os.path.exists(script_path):
                os.unlink(script_path)
