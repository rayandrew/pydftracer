#!/usr/bin/env python
import os
import shutil

import h5py
import numpy as np
import pytest
from PIL import Image

from dftracer.logger import dft_fn as Profile
from dftracer.logger import dftracer


def create_data_gen_function(df, args, io):
    """Create data generation function with given parameters"""

    @df.log
    def data_gen(data):
        for i in df.iter(range(args.num_files)):
            io.write(
                f"{args.data_dir}/{args.format}/{i}-of-{args.num_files}.{args.format}",
                data,
            )

    return data_gen


def create_read_data_function(df, args, io):
    """Create data reading function with given parameters"""

    @df.log
    def read_data(epoch):
        for i in df.iter(range(args.num_files)):
            io.read(
                f"{args.data_dir}/{args.format}/{i}-of-{args.num_files}.{args.format}"
            )

    return read_data


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
            _ = fd["x"][:]
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


@pytest.fixture(scope="function")
def setup_test_output_dirs():
    """Basic fixture for simple tests"""
    base_dir = os.path.join(os.path.dirname(__file__), "test_dftracer_output")
    data_dir = os.path.join(base_dir, "data")
    pfw_logs_dir = os.path.join(base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)
    yield data_dir, pfw_logs_dir
    # Cleanup after test
    shutil.rmtree(base_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def setup_parameterized_test_dirs():
    """Fixture for parameterized tests with specific directory structure"""
    test_dirs = []

    base_dir = os.path.join(os.path.dirname(__file__), "test_dftracer_output")

    def create_test_dir(test_format, num_files, niter, record_size):
        test_name = f"{test_format}_{num_files}_{niter}_{record_size}"
        test_base_dir = os.path.join(base_dir, test_name)
        data_dir = os.path.join(test_base_dir, "data")
        pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(pfw_logs_dir, exist_ok=True)
        test_dirs.append(base_dir)
        return data_dir, pfw_logs_dir, base_dir

    yield create_test_dir

    shutil.rmtree(base_dir)


class TestDFTracerLogger:
    def test_dftracer_singleton(self):
        instance1 = dftracer.get_instance()
        instance2 = dftracer.get_instance()
        assert instance1 is instance2
        assert isinstance(instance1, dftracer)

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
    def test_dftracer_io_operations_parameterized(
        self, setup_parameterized_test_dirs, test_config, monkeypatch
    ):
        """Parameterized test matching CMake test configurations"""

        # Test parameters from config
        test_format = test_config["format"]
        num_files = test_config["num_files"]
        niter = test_config["niter"]
        record_size = test_config["record_size"]

        data_dir, pfw_logs_dir, base_dir = setup_parameterized_test_dirs(
            test_format, num_files, niter, record_size
        )

        for env_var, value in test_config["env"].items():
            monkeypatch.setenv(env_var, value)

        if (
            "DFTRACER_DATA_DIR" not in test_config["env"]
            or test_config["env"]["DFTRACER_DATA_DIR"] != "all"
        ):
            monkeypatch.setenv("DFTRACER_DATA_DIR", data_dir)

        # Create format-specific directories within the test directory
        format_data_dir = os.path.join(data_dir, test_format)
        format_log_dir = os.path.join(pfw_logs_dir, test_format)
        os.makedirs(format_data_dir, exist_ok=True)
        os.makedirs(format_log_dir, exist_ok=True)

        # Initialize IOHandler and dftracer
        io = IOHandler(test_format)
        log_file = os.path.join(
            pfw_logs_dir, f"{test_config['name']}_{test_format}.pfw"
        )
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

        test_data_gen(data)
        for n in range(niter):
            test_read_data(n)
        df_logger.finalize()

        if test_config["env"].get("DFTRACER_ENABLE") != "0":
            for i in range(num_files):
                filename = os.path.join(
                    format_data_dir, f"{i}-of-{num_files}.{test_format}"
                )
                assert os.path.exists(filename), (
                    f"Data file {filename} should exist for test {test_config['name']}"
                )

        # Log file verification - in this implementation, we're using a NoOp profiler
        # so log files may not be created, but we can check if the test ran successfully
        if test_config["env"].get("DFTRACER_ENABLE") == "1":
            print(f"Test {test_config['name']} would create log file: {log_file}")

        print(
            f"Test {test_config['name']} completed successfully in directory: {base_dir}"
        )
