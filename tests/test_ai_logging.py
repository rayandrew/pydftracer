#!/usr/bin/env python
"""
AI Logging Test Suite for dftracer
"""

import os
import shutil
from dataclasses import dataclass
from time import sleep
from typing import Optional

import numpy as np
import pytest
from dftracer.python import ai, dftracer

from .utils import run_test_in_spawn_process, validate_log_files


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
        hook = Hook()
        df_logger = dftracer.initialize_log(
            logfile=log_file, data_dir=data_dir, process_id=-1
        )
        train(args, hook)
        df_logger.finalize()

        # Validate log files using the common utility
        expected_count = test_config.get("expected_events", 0)
        validate_log_files(log_file, test_config["name"], expected_count)
    finally:
        shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


class TestAILogging:
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "disabled",
                "num_files": 2,
                "niter": 1,
                "expected_events": 0,
                "env": {
                    "DFTRACER_ENABLE": "0",
                },
            },
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
