#!/usr/bin/env python
"""
Parallel Workers Test Suite for dftracer
"""

import os
import shutil
import threading
from multiprocessing import get_context
from time import sleep

import h5py
import numpy as np
import PIL.Image as im
import pytest
from dftracer.python import dft_fn, dftracer

from .utils import run_test_in_spawn_process


class IOHandler:
    def __init__(self, filename, format):
        self.format = format
        self.filename = filename


def posix_calls(val):
    data_dir, index, is_spawn = val
    path = f"{data_dir}/demofile{index}.txt"
    f = open(path, "w+")
    f.write("Now the file has more content!")
    f.close()
    if is_spawn:
        print(f"Calling spawn on {index} with pid {os.getpid()}")
    else:
        print(f"Not calling spawn on {index} with pid {os.getpid()}")


def npz_calls(data_dir, index):
    path = f"{data_dir}/demofile{index}.npz"
    if os.path.exists(path):
        os.remove(path)
    records = np.random.randint(255, size=(8, 8, 1024), dtype=np.uint8)
    record_labels = [0] * 1024
    np.savez(path, x=records, y=record_labels)


def jpeg_calls(data_dir, index):
    records = np.random.randint(255, size=(1024, 1024), dtype=np.uint8)
    img = im.fromarray(records)
    out_path_spec = f"{data_dir}/test.jpeg"
    img.save(out_path_spec, format="JPEG", bits=8)
    with open(out_path_spec, "rb") as f:
        image = im.open(f)
        _ = np.asarray(image)


def init():
    """This function is called when new processes start."""
    print(f"Initializing process {os.getpid()}")


def run_single_parallel_workers_test(test_config):
    base_dir = os.path.join(
        os.path.dirname(__file__), "test_parallel_workers_subprocess_output"
    )
    test_name = f"{test_config['name']}_files{test_config['num_files']}_niter{test_config['niter']}_format{test_config['format']}"
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    log_dir = os.path.join(test_base_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/{test_config['format']}", exist_ok=True)

    log_file = os.path.join(log_dir, f"{test_config['name']}.pfw")

    print(
        f"Running parallel workers test {test_config['name']} with log file: {log_file}"
    )

    log_inst = dftracer.initialize_log(
        logfile=log_file, data_dir=data_dir, process_id=-1
    )
    dft_fn_instance = dft_fn("COMPUTE")

    class IOReader(IOHandler):
        @dft_fn_instance.log_init
        def __init__(self, filename, format):
            super().__init__(filename=filename, format=format)
            if self.format == "jpeg" or self.format == "png":
                self.data = np.asarray(im.open(filename))
            if self.format == "npz":
                self.data = np.load(filename)
            if self.format == "hdf5":
                fd = h5py.File(filename, "r")
                self.data = fd["x"][:]  # type: ignore
                fd.close()

        def get(self):
            return self.data

    class IOWriter(IOHandler):
        @dft_fn_instance.log_init
        def __init__(self, filename, format, data):
            super().__init__(filename=filename, format=format)
            if self.format == "jpeg" or self.format == "png":
                image = im.fromarray(data)
                image.save(filename)
            if self.format == "npz":
                with open(filename, "wb") as f:
                    np.save(f, data)
            if self.format == "hdf5":
                fd = h5py.File(filename, "w")
                fd.create_dataset("x", data=data)
                fd.close()

    def custom_events():
        log_inst.enter_event()
        start = log_inst.get_time()
        sleep(0.001)
        end = log_inst.get_time()
        log_inst.log_event("test", "cat2", start, end - start)
        log_inst.exit_event()
        for _ in dft_fn_instance.iter(range(2)):
            sleep(0.001)

    @dft_fn_instance.log
    def log_events(index):
        sleep(0.001)

    @dft_fn_instance.log
    def data_gen(num_files, data_dir, format, data):
        for i in dft_fn_instance.iter(range(num_files)):
            IOWriter(
                filename=f"{data_dir}/{format}/{i}-of-{num_files}.{format}",
                format=format,
                data=data,
            )

    @dft_fn_instance.log
    def read_data(num_files, data_dir, format):
        for i in dft_fn_instance.iter(range(num_files)):
            io = IOReader(
                filename=f"{data_dir}/{format}/{i}-of-{num_files}.{format}",
                format=format,
            )
            _ = io.get()

    @dft_fn_instance.log
    def with_default_args(step=2):
        for i in dft_fn_instance.iter(range(step)):
            print(i)

    try:
        dft_fn_instance.log_metadata("key", "value")
        posix_calls((data_dir, 20, False))

        t1 = threading.Thread(target=posix_calls, args=((data_dir, 10, False),))
        custom_events()
        t2 = threading.Thread(
            target=npz_calls,
            args=(
                data_dir,
                1,
            ),
        )
        t3 = threading.Thread(
            target=jpeg_calls,
            args=(
                data_dir,
                2,
            ),
        )
        t4 = threading.Thread(target=log_events, args=(3,))

        t1.start()
        t2.start()
        t3.start()
        t4.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()

        index = 4
        with get_context("fork").Pool(1, initializer=init) as pool:
            pool.map(posix_calls, ((data_dir, index, False),))
        index = index + 1

        with get_context("spawn").Pool(1, initializer=init) as pool:
            pool.map(posix_calls, ((data_dir, index, True),))
        index = index + 1

        data = np.ones((test_config["record_size"], 1), dtype=np.uint8)
        data_gen(
            num_files=test_config["num_files"],
            data_dir=data_dir,
            format=test_config["format"],
            data=data,
        )

        for _ in range(test_config["niter"]):
            read_data(
                num_files=test_config["num_files"],
                data_dir=data_dir,
                format=test_config["format"],
            )

        with_default_args()
        log_inst.finalize()

    finally:
        shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


class TestParallelWorkers:
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "parallel_workers_npz",
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 1024,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                },
            },
            {
                "name": "parallel_workers_jpeg",
                "format": "jpeg",
                "num_files": 1,
                "niter": 1,
                "record_size": 512,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                },
            },
            {
                "name": "parallel_workers_hdf5",
                "format": "hdf5",
                "num_files": 1,
                "niter": 1,
                "record_size": 256,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                },
            },
        ],
    )
    def test_parallel_workers(self, test_config):
        run_test_in_spawn_process(run_single_parallel_workers_test, test_config)

    def test_parallel_workers_disabled(self):
        test_config = {
            "name": "parallel_workers_disabled",
            "format": "npz",
            "num_files": 1,
            "niter": 1,
            "record_size": 256,
            "env": {
                "DFTRACER_ENABLE": "0",
            },
        }
        run_test_in_spawn_process(run_single_parallel_workers_test, test_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
