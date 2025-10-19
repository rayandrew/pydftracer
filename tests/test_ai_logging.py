#!/usr/bin/env python
"""
AI Logging Test Suite for dftracer
"""

import glob
import os
import shutil
from dataclasses import dataclass
from time import sleep
from typing import Optional

import numpy as np
import pytest
from dftracer.python import DFTRACER_ENABLE, ai, dftracer

from .utils import run_test_in_spawn_process


@dataclass
class Args:
    log_dir: str
    data_dir: str
    num_files: int
    niter: int
    disable_ai_cat: Optional[str] = None
    epoch_as_metadata: bool = False
    record_size: int = 1048576


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


def run_ai_logging_test(test_config):
    base_dir = os.path.join(os.path.dirname(__file__), "test_ai_logging_output")
    test_name = f"{test_config['name']}_niter{test_config['niter']}_files{test_config['num_files']}"
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    log_dir = os.path.join(test_base_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{test_config['name']}.pfw")

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
        num_files=int(test_config["num_files"]),
        niter=int(test_config["niter"]),
        epoch_as_metadata=test_config.get("epoch_as_metadata", False),
        record_size=test_config.get("record_size", 1048576),
    )

    print(
        f"Running AI logging test {test_config['name']} with log file: {log_file}, args = {args}"
    )

    try:
        cpp_library_available = True
        hook = Hook()
        df_logger = dftracer.initialize_log(
            logfile=log_file, data_dir=data_dir, process_id=-1
        )
        train(args, hook)
        df_logger.finalize()

        event_count = 0
        if DFTRACER_ENABLE:
            log_pattern = log_file.replace(".pfw", "*-app.pfw")
            log_files = glob.glob(log_pattern)

            if not log_files and os.path.exists(log_file):
                log_files = [log_file]

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

            if event_count == 0:
                log_dir = os.path.dirname(log_file)
                if os.path.exists(log_dir):
                    log_dir_files = os.listdir(log_dir)
                    print(f"Files in log directory: {log_dir_files}")
                else:
                    print(f"Log directory does not exist: {log_dir}")

                try:
                    import dftracer.dftracer as cpp_libs  # noqa: F401

                    print("dftracer C++ library is available")
                except ImportError:
                    cpp_library_available = False
                    print(
                        "dftracer C++ library is NOT available - tests will run in no-op mode"
                    )

        print(f"Test {test_config['name']} completed with {event_count} events")

        expected_count = test_config.get("expected_events", 0)
        print(f"Expected event count: {expected_count}")
        if not DFTRACER_ENABLE:
            assert event_count == 0, (
                f"Expected 0 events when DFTRACER_ENABLE=0 but got {event_count} for test {test_config['name']}"
            )
        elif not cpp_library_available:
            assert event_count == 0, (
                f"Expected 0 events when C++ library not available but got {event_count} for test {test_config['name']}"
            )
        elif expected_count > 0:
            assert event_count == expected_count, (
                f"Expected {expected_count} events but got {event_count} for test {test_config['name']}"
            )
        else:
            assert event_count > 5, (
                f"Expected some events but got {event_count} for test {test_config['name']}"
            )
    finally:
        shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


class TestAILogging:
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "normal",
                "num_files": 2,
                "niter": 3,
                "expected_events": 76,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "epoch_as_metadata",
                "num_files": 2,
                "niter": 3,
                "epoch_as_metadata": True,
                "expected_events": 76,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_all",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "all",
                "expected_events": 9,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_dataloader",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "dataloader",
                "expected_events": 61,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_device",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "device",
                "expected_events": 70,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_compute",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "compute",
                "expected_events": 52,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
            {
                "name": "disable_cat_ckpt",
                "num_files": 2,
                "niter": 3,
                "disable_ai_cat": "ckpt",
                "expected_events": 73,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                    "DFTRACER_TRACE_COMPRESSION": "0",
                    "DFTRACER_DISABLE_IO": "1",
                },
            },
        ],
    )
    def test_ai_logging(self, test_config):
        run_test_in_spawn_process(run_ai_logging_test, test_config)

    def test_ai_logging_disabled(self):
        test_config = {
            "name": "disabled",
            "num_files": 2,
            "niter": 1,
            "expected_events": 0,
            "env": {
                "DFTRACER_ENABLE": "0",
            },
        }
        run_test_in_spawn_process(run_ai_logging_test, test_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
