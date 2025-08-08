#!/usr/bin/env python
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pytest
from PIL import Image

from dftracer.logger import dft_fn as Profile
from dftracer.logger import dftracer


def run_single_dftracer_test(test_config):
    """Run a single dftracer test in isolation - suitable for subprocess execution"""
    test_format = test_config["format"]
    num_files = test_config["num_files"]
    niter = test_config["niter"]
    record_size = test_config["record_size"]

    # Set environment variables
    for env_var, value in test_config["env"].items():
        os.environ[env_var] = value

    # Create test directories with unique names per test
    base_dir = os.path.join(os.path.dirname(__file__), "test_dftracer_output")
    test_name = f"{test_config['name']}_{test_format}_{num_files}_{niter}_{record_size}"
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)

    if (
        "DFTRACER_DATA_DIR" not in test_config["env"]
        or test_config["env"]["DFTRACER_DATA_DIR"] != "all"
    ):
        os.environ["DFTRACER_DATA_DIR"] = data_dir

    # Create format-specific directories
    format_data_dir = os.path.join(data_dir, test_format)
    format_log_dir = os.path.join(pfw_logs_dir, test_format)
    os.makedirs(format_data_dir, exist_ok=True)
    os.makedirs(format_log_dir, exist_ok=True)

    # Initialize IOHandler and dftracer
    io = IOHandler(test_format)
    log_file = os.path.join(pfw_logs_dir, f"{test_config['name']}_{test_format}.pfw")
    print(f"Running test {test_config['name']} with log file: {log_file}")

    df_logger = dftracer.initialize_log(log_file, data_dir, -1)

    data = np.ones((record_size, 1), dtype=np.uint8)
    df_test = Profile(f"dft_{test_config['name']}")

    @df_test.log
    def test_data_gen(test_data):
        for i in df_test.iter(range(num_files)):
            filename = os.path.join(
                format_data_dir, f"{i}-of-{num_files}.{test_format}"
            )
            io.write(filename, test_data)

    @df_test.log
    def test_read_data(epoch):
        for i in df_test.iter(range(num_files)):
            filename = os.path.join(
                format_data_dir, f"{i}-of-{num_files}.{test_format}"
            )
            io.read(filename)

    # Execute the test workflow
    test_data_gen(data)
    for n in range(niter):
        test_read_data(n)
    df_logger.finalize()

    # Verify files were created (except when DFTRACER is disabled)
    if test_config["env"].get("DFTRACER_ENABLE") != "0":
        for i in range(num_files):
            filename = os.path.join(
                format_data_dir, f"{i}-of-{num_files}.{test_format}"
            )
            assert os.path.exists(filename), (
                f"Data file {filename} should exist for test {test_config['name']}"
            )

    print(f"Test {test_config['name']} completed successfully")

    # shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


class IOHandler:
    def __init__(self, format):
        self.format = format

    def read(self, filename):
        if self.format == "jpeg" or self.format == "png":
            return np.asarray(Image.open(filename))
        if self.format == "npz":
            return np.load(filename)
        if self.format == "hdf5":
            fd = h5py.File(filename, "r")
            _ = fd["x"][:]  # type: ignore
            fd.close()

    def write(self, filename, a):
        if self.format == "jpeg" or self.format == "png":
            im = Image.fromarray(a)
            # im.show()
            im.save(filename)
        if self.format == "npz":
            with open(filename, "wb") as f:
                np.save(f, a)
        if self.format == "hdf5":
            fd = h5py.File(filename, "w")
            fd.create_dataset("x", data=a)
            fd.close()


class TestDFTracerLogger:
    @pytest.mark.subprocess
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "disable_only",
                "env": {"DFTRACER_ENABLE": "0"},
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 256,
            },
            {
                "name": "disable_io_only",
                "env": {"DFTRACER_ENABLE": "1", "DFTRACER_DISABLE_IO": "1"},
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 512,
            },
            {
                "name": "io_all_only",
                "env": {"DFTRACER_ENABLE": "1", "DFTRACER_DATA_DIR": "all"},
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 1024,
            },
            {
                "name": "io_specific_only",
                "env": {"DFTRACER_ENABLE": "1"},
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 2048,
            },
            {
                "name": "io_specific_meta_only",
                "env": {"DFTRACER_ENABLE": "1", "DFTRACER_INC_METADATA": "1"},
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 4096,
            },
            {
                "name": "io_all_meta_only",
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_DATA_DIR": "all",
                },
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 2048,
            },
        ],
    )
    def test_dftracer(self, test_config):
        """Run each dftracer test configuration in a separate subprocess"""

        # Create a temporary Python script that runs the test
        script_content = f'''
import sys
import os
sys.path.insert(0, "{os.path.dirname(os.path.dirname(__file__))}")

from tests.test_dftracer import run_single_dftracer_test

test_config = {test_config!r}

if __name__ == "__main__":
    try:
        result = run_single_dftracer_test(test_config)
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test failed with error: {{e}}")
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
                timeout=60,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
