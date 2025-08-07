#!/usr/bin/env python
"""
Parallel Workers Test Suite for dftracer
"""

import os
import shutil
import subprocess
import tempfile
import threading
from multiprocessing import get_context
from time import sleep

import h5py
import numpy as np
import PIL.Image as im
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
    # Create test directories with unique names per test
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

    print(
        f"Running parallel workers test {test_config['name']} with log file: {log_file}"
    )

    # Initialize dftracer

    from dftracer.logger import dft_fn, dftracer

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

    # Execute the test workflow
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

        # Start all threads
        t1.start()
        t2.start()
        t3.start()
        t4.start()

        # Wait for all threads to complete
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

        # Testing named parameters
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
    @pytest.mark.subprocess
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "parallel_workers_npz",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "format": "npz",
                "num_files": 2,
                "niter": 1,
                "record_size": 1024,
            },
            {
                "name": "parallel_workers_jpeg",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "format": "jpeg",
                "num_files": 1,
                "niter": 1,
                "record_size": 512,
            },
            {
                "name": "parallel_workers_hdf5",
                "env": {
                    "DFTRACER_INIT": "PRELOAD",
                    "LD_PRELOAD": get_dftracer_preload_path(),
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_ENABLE": "1",
                },
                "format": "hdf5",
                "num_files": 1,
                "niter": 1,
                "record_size": 256,
            },
            {
                "name": "parallel_workers_disabled",
                "env": {
                    "DFTRACER_ENABLE": "0",
                },
                "format": "npz",
                "num_files": 1,
                "niter": 1,
                "record_size": 256,
            },
        ],
    )
    def test_parallel_workers(self, test_config):
        """Run each parallel workers test configuration in a separate subprocess"""

        # Create a temporary Python script that runs the test
        test_script_template = f'''
import sys
import os
sys.path.insert(0, "{os.path.dirname(os.path.dirname(__file__))}")

from tests.test_parallel_workers import run_single_parallel_workers_test

test_config = {test_config!r}

if __name__ == "__main__":
    try:
        result = run_single_parallel_workers_test(test_config)
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test failed with error: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_script_template)
            f.flush()
            script_path = f.name

            try:
                # Run the test script in subprocess
                result = subprocess.run(
                    ["python", script_path], capture_output=True, text=True, timeout=120
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
