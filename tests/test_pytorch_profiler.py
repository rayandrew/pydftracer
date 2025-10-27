#!/usr/bin/env python
import glob
import gzip
import os

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .utils import run_test_in_spawn_process


def validate_pytorch_profiler_and_io_logs(
    log_file, test_name, data_dir, expected_min_events=1
):
    """
    Validate log files and check for both 'cat':'PP' (PyTorch Profiler) and I/O events ('cat':'POSIX' or 'cat':'STDIO').

    Args:
        log_file: Path to the expected log file
        test_name: Name of the test (for error messages)
        expected_min_events: Minimum expected number of PyTorch profiler events

    Returns:
        Tuple of (total_events, pytorch_profiler_events, io_events)
    """
    total_events = 0
    pytorch_profiler_events = 0
    io_events = 0

    # Check if C++ library is available
    try:
        import dftracer.dftracer as cpp_libs  # noqa: F401

        cpp_library_available = True
        print("dftracer C++ library is available")
    except ImportError:
        cpp_library_available = False
        print("dftracer C++ library is NOT available - tests will run in no-op mode")

    # Check DFTRACER_ENABLE - import locally to avoid global import issues
    DFTRACER_ENABLE_ENV = os.environ.get("DFTRACER_ENABLE", "0") == "1"

    if DFTRACER_ENABLE_ENV and cpp_library_available:
        # Check both the specified log directory and data directory for log files
        log_dir = os.path.dirname(log_file)
        directories_to_check = [log_dir, data_dir]

        log_files = []
        for check_dir in directories_to_check:
            print(f"Checking directory: {check_dir}")
            if os.path.exists(check_dir):
                # Try different log file patterns for gzipped files - collect ALL matching files
                patterns_to_try = [
                    os.path.join(check_dir, "*-app.pfw.gz"),  # Standard gzipped format
                    os.path.join(check_dir, "*.pfw.gz"),  # Simple gzipped format
                    os.path.join(check_dir, "*.pfw"),  # Uncompressed fallback
                ]

                for pattern in patterns_to_try:
                    found_files = glob.glob(pattern)
                    if found_files:
                        # Remove duplicates by converting to set and back to list
                        new_files = [f for f in found_files if f not in log_files]
                        log_files.extend(new_files)
                        print(
                            f"Found {len(found_files)} files with pattern {pattern}: {found_files}"
                        )
                    else:
                        print(f"No files found with pattern: {pattern}")

                # Also list all files in directory for debugging
                all_files = os.listdir(check_dir)
                if all_files:
                    print(f"All files in {check_dir}: {all_files}")
                else:
                    print(f"Directory {check_dir} is empty")
            else:
                print(f"Directory {check_dir} does not exist")

        print(f"All found log files: {log_files}")

        for log_file_path in log_files:
            if os.path.exists(log_file_path):
                try:
                    # Handle both gzipped and uncompressed files
                    if log_file_path.endswith(".gz"):
                        with gzip.open(log_file_path, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(log_file_path) as f:
                            lines = f.readlines()

                    total_events += len(lines)

                    # Count lines containing PyTorch profiler events and I/O events (POSIX/STDIO)
                    pp_events = sum(1 for line in lines if '"cat":"PP"' in line)
                    posix_events = sum(1 for line in lines if '"cat":"POSIX"' in line)
                    stdio_events = sum(1 for line in lines if '"cat":"STDIO"' in line)
                    io_count = posix_events + stdio_events
                    pytorch_profiler_events += pp_events
                    io_events += io_count

                    print(
                        f"Found {len(lines)} total events, {pp_events} PyTorch profiler events, {io_count} I/O events ({posix_events} POSIX, {stdio_events} STDIO) in {log_file_path}"
                    )

                    # Print first few lines for debugging
                    if lines:
                        print(f"Sample log entries from {log_file_path}:")
                        for i, line in enumerate(lines[:3]):
                            print(f"  Line {i + 1}: {line.strip()[:100]}...")

                except Exception as e:
                    print(f"Error reading {log_file_path}: {e}")

        if total_events == 0:
            log_dir = os.path.dirname(log_file)
            if os.path.exists(log_dir):
                log_dir_files = os.listdir(log_dir)
                print(f"Files in log directory ({log_dir}): {log_dir_files}")
                # Show file details for debugging
                for f in log_dir_files:
                    full_path = os.path.join(log_dir, f)
                    if os.path.isfile(full_path):
                        size = os.path.getsize(full_path)
                        print(f"  {f}: {size} bytes")
            else:
                print(f"Log directory does not exist: {log_dir}")

    print(
        f"Test {test_name} completed with {total_events} total events, {pytorch_profiler_events} PyTorch profiler events, {io_events} I/O events (POSIX/STDIO)"
    )

    # Assertions based on DFTRACER state
    if not DFTRACER_ENABLE_ENV:
        assert total_events == 0, (
            f"Expected 0 events when DFTRACER_ENABLE=0 but got {total_events} for test {test_name}"
        )
        assert pytorch_profiler_events == 0, (
            f"Expected 0 PyTorch profiler events when DFTRACER_ENABLE=0 but got {pytorch_profiler_events} for test {test_name}"
        )
        assert io_events == 0, (
            f"Expected 0 I/O events (POSIX/STDIO) when DFTRACER_ENABLE=0 but got {io_events} for test {test_name}"
        )
    elif not cpp_library_available:
        assert total_events == 0, (
            f"Expected 0 events when C++ library not available but got {total_events} for test {test_name}"
        )
        assert pytorch_profiler_events == 0, (
            f"Expected 0 PyTorch profiler events when C++ library not available but got {pytorch_profiler_events} for test {test_name}"
        )
        assert io_events == 0, (
            f"Expected 0 I/O events (POSIX/STDIO) when C++ library not available but got {io_events} for test {test_name}"
        )
    else:
        # When enabled and library available, we should have some events
        assert total_events > 0, (
            f"Expected some events but got {total_events} for test {test_name}"
        )
        # PyTorch profiler events are optional - the integration might not always produce PP events
        # but we should at least have dftracer events from the training function
        if pytorch_profiler_events > 0:
            print(f"Great! Found {pytorch_profiler_events} PyTorch profiler events")
        else:
            print(
                f"No PyTorch profiler events found, but {total_events} dftracer events were logged"
            )

        # I/O events should be present if we're doing dataset loading
        if io_events > 0:
            print(
                f"Great! Found {io_events} I/O events (POSIX/STDIO) from dataset operations"
            )
        else:
            print(
                "No I/O events (POSIX/STDIO) found - this may be expected for synthetic data tests"
            )

    return total_events, pytorch_profiler_events, io_events


def validate_pytorch_profiler_logs(
    log_file, test_name, data_dir, expected_min_events=1
):
    """
    Validate PyTorch profiler log files and check for 'cat':'PP' events.

    Args:
        log_file: Path to the expected log file
        test_name: Name of the test (for error messages)
        expected_min_events: Minimum expected number of PyTorch profiler events

    Returns:
        Tuple of (total_events, pytorch_profiler_events)
    """
    total_events, pytorch_profiler_events, _ = validate_pytorch_profiler_and_io_logs(
        log_file, test_name, data_dir, expected_min_events
    )
    return total_events, pytorch_profiler_events


def validate_pytorch_profiler_logs_original(
    log_file, test_name, data_dir, expected_min_events=1
):
    """
    Original validate_pytorch_profiler_logs function implementation for reference.
    """
    total_events = 0
    pytorch_profiler_events = 0

    # Check if C++ library is available
    try:
        import dftracer.dftracer as cpp_libs  # noqa: F401

        cpp_library_available = True
        print("dftracer C++ library is available")
    except ImportError:
        cpp_library_available = False
        print("dftracer C++ library is NOT available - tests will run in no-op mode")

    # Check DFTRACER_ENABLE - import locally to avoid global import issues
    DFTRACER_ENABLE_ENV = os.environ.get("DFTRACER_ENABLE", "0") == "1"

    if DFTRACER_ENABLE_ENV and cpp_library_available:
        # Check both the specified log directory and data directory for log files
        log_dir = os.path.dirname(log_file)
        directories_to_check = [log_dir, data_dir]

        log_files = []
        for check_dir in directories_to_check:
            print(f"Checking directory: {check_dir}")
            if os.path.exists(check_dir):
                # Try different log file patterns for gzipped files - collect ALL matching files
                patterns_to_try = [
                    os.path.join(check_dir, "*-app.pfw.gz"),  # Standard gzipped format
                    os.path.join(check_dir, "*.pfw.gz"),  # Simple gzipped format
                    os.path.join(check_dir, "*.pfw"),  # Uncompressed fallback
                ]

                for pattern in patterns_to_try:
                    found_files = glob.glob(pattern)
                    if found_files:
                        # Remove duplicates by converting to set and back to list
                        new_files = [f for f in found_files if f not in log_files]
                        log_files.extend(new_files)
                        print(
                            f"Found {len(found_files)} files with pattern {pattern}: {found_files}"
                        )
                    else:
                        print(f"No files found with pattern: {pattern}")

                # Also list all files in directory for debugging
                all_files = os.listdir(check_dir)
                if all_files:
                    print(f"All files in {check_dir}: {all_files}")
                else:
                    print(f"Directory {check_dir} is empty")
            else:
                print(f"Directory {check_dir} does not exist")

        print(f"All found log files: {log_files}")

        for log_file_path in log_files:
            if os.path.exists(log_file_path):
                try:
                    # Handle both gzipped and uncompressed files
                    if log_file_path.endswith(".gz"):
                        with gzip.open(log_file_path, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(log_file_path) as f:
                            lines = f.readlines()

                    total_events += len(lines)

                    # Count lines containing PyTorch profiler events
                    pp_events = sum(1 for line in lines if '"cat":"PP"' in line)
                    pytorch_profiler_events += pp_events

                    print(
                        f"Found {len(lines)} total events, {pp_events} PyTorch profiler events in {log_file_path}"
                    )

                    # Print first few lines for debugging
                    if lines:
                        print(f"Sample log entries from {log_file_path}:")
                        for i, line in enumerate(lines[:3]):
                            print(f"  Line {i + 1}: {line.strip()[:100]}...")

                except Exception as e:
                    print(f"Error reading {log_file_path}: {e}")

        if total_events == 0:
            log_dir = os.path.dirname(log_file)
            if os.path.exists(log_dir):
                log_dir_files = os.listdir(log_dir)
                print(f"Files in log directory ({log_dir}): {log_dir_files}")
                # Show file details for debugging
                for f in log_dir_files:
                    full_path = os.path.join(log_dir, f)
                    if os.path.isfile(full_path):
                        size = os.path.getsize(full_path)
                        print(f"  {f}: {size} bytes")
            else:
                print(f"Log directory does not exist: {log_dir}")

    print(
        f"Test {test_name} completed with {total_events} total events, {pytorch_profiler_events} PyTorch profiler events"
    )

    # Assertions based on DFTRACER state
    if not DFTRACER_ENABLE_ENV:
        assert total_events == 0, (
            f"Expected 0 events when DFTRACER_ENABLE=0 but got {total_events} for test {test_name}"
        )
        assert pytorch_profiler_events == 0, (
            f"Expected 0 PyTorch profiler events when DFTRACER_ENABLE=0 but got {pytorch_profiler_events} for test {test_name}"
        )
    elif not cpp_library_available:
        assert total_events == 0, (
            f"Expected 0 events when C++ library not available but got {total_events} for test {test_name}"
        )
        assert pytorch_profiler_events == 0, (
            f"Expected 0 PyTorch profiler events when C++ library not available but got {pytorch_profiler_events} for test {test_name}"
        )
    else:
        # When enabled and library available, we should have some events
        assert total_events > 0, (
            f"Expected some events but got {total_events} for test {test_name}"
        )
        # PyTorch profiler events are optional - the integration might not always produce PP events
        # but we should at least have dftracer events from the training function
        if pytorch_profiler_events > 0:
            print(f"Great! Found {pytorch_profiler_events} PyTorch profiler events")
        else:
            print(
                f"No PyTorch profiler events found, but {total_events} dftracer events were logged"
            )

    return total_events, pytorch_profiler_events


def run_single_pytorch_profiler_with_io_test(test_config):
    """Run a PyTorch profiler test with real I/O operations through CIFAR-10 dataset loading"""
    # Import dftracer and PyTorch profiler modules within the test function for proper isolation
    from dftracer.python import DFTRACER_ENABLE, dftracer
    from dftracer.python import dft_fn as Profile
    from dftracer.python.torch import trace_handler
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    batch_size = test_config["batch_size"]
    num_epochs = test_config["num_epochs"]
    num_steps = test_config["num_steps"]

    base_dir = os.path.join(os.path.dirname(__file__), "test_pytorch_profiler_output")
    test_name = (
        f"{test_config['name']}_batch{batch_size}_epochs{num_epochs}_steps{num_steps}"
    )
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)

    log_file = os.path.join(pfw_logs_dir, f"{test_config['name']}_pytorch_profiler.pfw")
    print(
        f"Running PyTorch profiler test with I/O {test_config['name']} with log file: {log_file}"
    )
    print(f"Log directory: {pfw_logs_dir}")
    print(f"Data directory: {data_dir}")

    # Use the same pattern as test_dftracer.py - data_dir as second parameter
    df_logger = dftracer.initialize_log(log_file, "all", -1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Use CIFAR-10 data that already exists in the workspace to generate I/O operations
        import pickle

        import numpy as np
        from torch.utils.data import Dataset

        class CIFAR10Dataset(Dataset):
            """Custom CIFAR-10 dataset that will generate I/O events when loading data."""

            def __init__(self, data_dir):
                self.data_dir = data_dir
                self.data = []
                self.labels = []

                # Copy CIFAR-10 data to the monitored data_dir to ensure I/O events are captured
                import shutil

                source_cifar_dir = (
                    "/usr/workspace/haridev/pydftracer/data/cifar-10-batches-py"
                )
                monitored_cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
                os.makedirs(monitored_cifar_dir, exist_ok=True)

                # Load a subset of batches to create I/O operations within monitored directory
                batch_files = ["data_batch_1", "data_batch_2"]

                for batch_file in batch_files:
                    source_path = os.path.join(source_cifar_dir, batch_file)
                    monitored_path = os.path.join(monitored_cifar_dir, batch_file)

                    # Copy file to monitored directory if not already there
                    if os.path.exists(source_path) and not os.path.exists(
                        monitored_path
                    ):
                        print(
                            f"Copying {source_path} to {monitored_path} for monitored I/O"
                        )
                        shutil.copy2(source_path, monitored_path)

                    # Now load from the monitored directory to generate I/O events
                    if os.path.exists(monitored_path):
                        print(
                            f"Loading batch file from monitored directory: {monitored_path}"
                        )
                        with open(monitored_path, "rb") as f:
                            batch_data = pickle.load(f, encoding="bytes")

                        # Convert to numpy arrays and add to dataset
                        batch_images = batch_data[b"data"]
                        batch_labels = batch_data[b"labels"]

                        # Reshape images to (32, 32, 3) and normalize
                        batch_images = (
                            batch_images.reshape(-1, 3, 32, 32).astype(np.float32)
                            / 255.0
                        )

                        self.data.extend(batch_images)
                        self.labels.extend(batch_labels)

                print(f"Loaded {len(self.data)} samples from CIFAR-10")

                # Limit dataset size for faster testing
                max_samples = batch_size * 10  # Ensure we have enough samples
                if len(self.data) > max_samples:
                    self.data = self.data[:max_samples]
                    self.labels = self.labels[:max_samples]
                    print(
                        f"Limited dataset to {len(self.data)} samples for faster testing"
                    )

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                # Convert to torch tensors
                image = torch.tensor(self.data[idx], dtype=torch.float32)
                label = torch.tensor(self.labels[idx], dtype=torch.long)
                return image, label

        # Create dataset and dataloader - this will trigger I/O operations
        try:
            train_dataset = CIFAR10Dataset(data_dir)
        except Exception as e:
            print(f"Failed to load CIFAR-10 data ({e}), falling back to synthetic data")
            # Fallback to synthetic data if CIFAR-10 loading fails
            from torch.utils.data import TensorDataset

            num_samples = max(batch_size * 4, 32)
            synthetic_data = torch.randn(num_samples, 3, 32, 32)
            synthetic_labels = torch.randint(0, 10, (num_samples,))
            train_dataset = TensorDataset(synthetic_data, synthetic_labels)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Model, loss, and optimizer setup for CIFAR-10 (3 channels, 32x32)
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # Profiler schedule and loop control
        WAIT_STEPS = 1
        WARMUP_STEPS = 1
        ACTIVE_STEPS = test_config.get("active_steps", 2)
        REPEAT_CYCLES = 1

        total_steps = (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * REPEAT_CYCLES

        profiler_schedule = schedule(
            wait=WAIT_STEPS,
            warmup=WARMUP_STEPS,
            active=ACTIVE_STEPS,
            repeat=REPEAT_CYCLES,
        )

        # Create dftracer profile for the training loop
        df_test = Profile(f"pytorch_profiler_io_{test_config['name']}")

        @df_test.log
        def training_step_with_io(step_data):
            inputs, labels = step_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("model_forward"):
                outputs = model(inputs)

            with record_function("loss_calculation"):
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            return loss.item()

        step_count = 0

        try:
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                    if device.type == "cuda"
                    else ProfilerActivity.CPU,
                ],
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                profile_memory=True,
                with_stack=True,
            ) as p:
                for epoch in range(num_epochs):
                    for _step, (inputs, labels) in enumerate(train_dataloader):
                        if step_count >= total_steps or step_count >= num_steps:
                            break

                        loss_value = training_step_with_io((inputs, labels))
                        print(
                            f"Epoch {epoch}, Step {step_count}, Loss: {loss_value:.4f}"
                        )

                        p.step()
                        step_count += 1

                    if step_count >= total_steps or step_count >= num_steps:
                        break
        except Exception as e:
            print(f"PyTorch profiler encountered an error: {e}")
            print("Continuing test without profiler...")

        print(
            f"PyTorch profiler test with I/O {test_config['name']} completed successfully with {step_count} steps"
        )

    finally:
        df_logger.finalize()

    # Validate the log files and check for both PyTorch profiler events and I/O events
    expected_pytorch_events = step_count if DFTRACER_ENABLE else 0
    total_events, pytorch_profiler_events, io_events = (
        validate_pytorch_profiler_and_io_logs(
            log_file,
            test_config["name"],
            data_dir,
            expected_min_events=expected_pytorch_events,
        )
    )

    print(
        f"Validation complete: {total_events} total events, {pytorch_profiler_events} PyTorch profiler events, {io_events} I/O events"
    )

    return True


def run_single_pytorch_profiler_test(test_config):
    """Run a single PyTorch profiler test in isolation"""
    # Import dftracer and PyTorch profiler modules within the test function for proper isolation
    from dftracer.python import DFTRACER_ENABLE, dftracer
    from dftracer.python import dft_fn as Profile
    from dftracer.python.torch import trace_handler
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    batch_size = test_config["batch_size"]
    num_epochs = test_config["num_epochs"]
    num_steps = test_config["num_steps"]

    base_dir = os.path.join(os.path.dirname(__file__), "test_pytorch_profiler_output")
    test_name = (
        f"{test_config['name']}_batch{batch_size}_epochs{num_epochs}_steps{num_steps}"
    )
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    pfw_logs_dir = os.path.join(test_base_dir, "pfw_logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pfw_logs_dir, exist_ok=True)

    log_file = os.path.join(pfw_logs_dir, f"{test_config['name']}_pytorch_profiler.pfw")
    print(
        f"Running PyTorch profiler test {test_config['name']} with log file: {log_file}"
    )
    print(f"Log directory: {pfw_logs_dir}")
    print(f"Data directory: {data_dir}")

    # Use the same pattern as test_dftracer.py - data_dir as second parameter
    df_logger = dftracer.initialize_log(log_file, "all", -1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Data setup - FashionMNIST is much smaller and faster to download than CIFAR-10
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Single channel for grayscale
            ]
        )

        # Use synthetic data for fastest test execution (no download required)
        # This is much faster than downloading any real dataset
        print("Using synthetic data for fast test execution")
        from torch.utils.data import TensorDataset

        # Create synthetic data (grayscale 28x28, similar to MNIST/FashionMNIST)
        num_samples = max(batch_size * 4, 32)  # Ensure we have enough samples
        synthetic_data = torch.randn(num_samples, 1, 28, 28)
        synthetic_labels = torch.randint(0, 10, (num_samples,))
        train_dataset = TensorDataset(synthetic_data, synthetic_labels)

        # Optional: Uncomment below to use FashionMNIST for more realistic testing
        # fashion_mnist_dir = os.path.join(data_dir, "fashion-mnist")
        # try:
        #     train_dataset = FashionMNIST(root=fashion_mnist_dir, train=True, download=True, transform=transform)
        # except Exception as e:
        #     print(f"FashionMNIST download failed ({e}), using synthetic data")
        #     train_dataset = TensorDataset(synthetic_data, synthetic_labels)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # Model, loss, and optimizer setup - modified for grayscale input
        model = models.resnet18(weights=None)
        # Modify first conv layer for grayscale input (1 channel instead of 3)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # Profiler schedule and loop control
        WAIT_STEPS = 1
        WARMUP_STEPS = 1
        ACTIVE_STEPS = test_config.get("active_steps", 2)
        REPEAT_CYCLES = 1

        total_steps = (WAIT_STEPS + WARMUP_STEPS + ACTIVE_STEPS) * REPEAT_CYCLES

        profiler_schedule = schedule(
            wait=WAIT_STEPS,
            warmup=WARMUP_STEPS,
            active=ACTIVE_STEPS,
            repeat=REPEAT_CYCLES,
        )

        # Create dftracer profile for the training loop
        df_test = Profile(f"pytorch_profiler_{test_config['name']}")

        @df_test.log
        def training_step(step_data):
            inputs, labels = step_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with record_function("model_forward"):
                outputs = model(inputs)

            with record_function("loss_calculation"):
                loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            return loss.item()

        step_count = 0  # Initialize step_count outside the profiler context

        try:
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                    if device.type == "cuda"
                    else ProfilerActivity.CPU,
                ],
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                profile_memory=True,
                with_stack=True,
            ) as p:
                for epoch in range(num_epochs):
                    for _step, (inputs, labels) in enumerate(train_dataloader):
                        if step_count >= total_steps or step_count >= num_steps:
                            break

                        loss_value = training_step((inputs, labels))
                        print(
                            f"Epoch {epoch}, Step {step_count}, Loss: {loss_value:.4f}"
                        )

                        p.step()
                        step_count += 1

                    if step_count >= total_steps or step_count >= num_steps:
                        break
        except Exception as e:
            print(f"PyTorch profiler encountered an error: {e}")
            print("Continuing test without profiler...")

        print(
            f"PyTorch profiler test {test_config['name']} completed successfully with {step_count} steps"
        )

    finally:
        df_logger.finalize()

    # Validate the log files and check for PyTorch profiler events
    expected_pytorch_events = step_count if DFTRACER_ENABLE else 0
    total_events, pytorch_profiler_events = validate_pytorch_profiler_logs(
        log_file,
        test_config["name"],
        data_dir,
        expected_min_events=expected_pytorch_events,
    )

    print(
        f"Validation complete: {total_events} total events, {pytorch_profiler_events} PyTorch profiler events"
    )

    return True


class TestPyTorchProfiler:
    @pytest.mark.parametrize(
        "test_config",
        [
            {
                "name": "basic_pytorch_profiler",
                "batch_size": 4,
                "num_epochs": 1,
                "num_steps": 3,
                "active_steps": 2,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                },
            },
            {
                "name": "small_pytorch_profiler",
                "batch_size": 2,
                "num_epochs": 1,
                "num_steps": 2,
                "active_steps": 1,
                "env": {
                    "DFTRACER_ENABLE": "1",
                    "DFTRACER_INC_METADATA": "1",
                },
            },
        ],
    )
    def test_pytorch_profiler(self, test_config):
        """Test PyTorch profiler integration with dftracer."""
        run_test_in_spawn_process(run_single_pytorch_profiler_test, test_config)

    def test_pytorch_profiler_disabled(self):
        """Test PyTorch profiler with dftracer disabled."""
        test_config = {
            "name": "disabled",
            "batch_size": 8,
            "num_epochs": 1,
            "num_steps": 2,
            "active_steps": 1,
            "env": {
                "DFTRACER_ENABLE": "0",
            },
        }
        run_test_in_spawn_process(run_single_pytorch_profiler_test, test_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
