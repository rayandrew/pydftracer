#!/usr/bin/env python
import os

import h5py
import numpy as np
import pytest
from dftracer.python import DFTRACER_ENABLE, dftracer
from dftracer.python import dft_fn as Profile
from PIL import Image

from .utils import run_test_in_spawn_process


def run_single_dftracer_test(test_config):
    """Run a single dftracer test in isolation"""
    test_format = test_config["format"]
    num_files = test_config["num_files"]
    niter = test_config["niter"]
    record_size = test_config["record_size"]

    base_dir = os.path.join(os.path.dirname(__file__), "test_dftracer_output")
    test_name = f"{test_config['name']}_{test_format}_{num_files}_{niter}_{record_size}"
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)

    format_data_dir = os.path.join(data_dir, test_format)
    format_log_dir = os.path.join(pfw_logs_dir, test_format)
    os.makedirs(format_data_dir, exist_ok=True)
    os.makedirs(format_log_dir, exist_ok=True)

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

    test_data_gen(data)
    for n in range(niter):
        test_read_data(n)
    df_logger.finalize()

    if DFTRACER_ENABLE:
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
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "basic_npz",
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 256,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                },
            },
            {
                "name": "basic_jpeg",
                "format": "jpeg",
                "num_files": 2,
                "niter": 1,
                "record_size": 512,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                },
            },
            {
                "name": "basic_hdf5",
                "format": "hdf5",
                "num_files": 2,
                "niter": 1,
                "record_size": 1024,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                },
            },
        ],
    )
    def test_dftracer(self, test_config):
        run_test_in_spawn_process(run_single_dftracer_test, test_config)

    def test_dftracer_disabled(self):
        test_config = {
            "name": "disabled",
            "format": "npz",
            "num_files": 1,
            "niter": 1,
            "record_size": 256,
            "env": {
                "DFTRACER_ENABLE": "0",
            },
        }
        run_test_in_spawn_process(run_single_dftracer_test, test_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
