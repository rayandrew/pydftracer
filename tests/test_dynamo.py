import os
import shutil

import pytest
from dftracer.python.dbg import dftracer, dynamo
from dftracer.python.dynamo import create_backend

from .utils import run_test_in_spawn_process, validate_log_files


def _check_torch_available():
    """Check if torch is available in the current environment."""
    try:
        import torch  # noqa: F401 # type: ignore

        return True
    except ImportError:
        return False


def run_single_dynamo_test(test_config):
    assert _check_torch_available(), "Torch is not available and this should be skipped"

    import torch  # noqa: F401 # type: ignore

    # Setup test directories
    base_dir = os.path.join(os.path.dirname(__file__), "test_dynamo_output")
    test_name = test_config.get("name", "dynamo_test")
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    log_dir = os.path.join(test_base_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{test_name}.pfw")

    try:
        # Initialize logger
        df_logger = dftracer.initialize_log(log_file, data_dir, -1)

        # Define and run the model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, 1)
                self.fc = torch.nn.Linear(16 * 15 * 15, 10)

            @dynamo.compile
            def forward(self, x):
                x = self.conv(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = SimpleModel()
        t_model: torch.nn.Module = model  # type: ignore
        sample = torch.randn(1, 3, 32, 32)
        output = t_model(sample)
        print(f"Model output shape: {output.shape}")
        df_logger.finalize()

        # Validate log files using the common utility
        expected_count = test_config.get("expected_count", 0)
        mode = test_config.get("mode", "exact")
        validate_log_files(log_file, test_name, expected_count, mode=mode)
    finally:
        # Cleanup test directories
        shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


def run_create_backend_test(test_config):
    """Test the create_backend function with torch.compile"""
    assert _check_torch_available(), "Torch is not available and this should be skipped"

    import torch  # noqa: F401 # type: ignore

    # Setup test directories
    base_dir = os.path.join(os.path.dirname(__file__), "test_dynamo_output")
    test_name = test_config.get("name", "create_backend_test")
    test_base_dir = os.path.join(base_dir, test_name)
    data_dir = os.path.join(test_base_dir, "data")
    log_dir = os.path.join(test_base_dir, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{test_name}.pfw")

    try:
        # Initialize logger
        df_logger = dftracer.initialize_log(log_file, data_dir, -1)

        # Create backend with custom parameters
        backend = create_backend(
            name="test_model",
            epoch=0,
            step=0,
            enable=True,
            autograd=test_config.get("autograd", True),
        )

        # Define a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, 1)
                self.fc = torch.nn.Linear(16 * 15 * 15, 10)

            def forward(self, x):
                x = self.conv(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = SimpleModel()
        # Use torch.compile with the custom backend
        compiled_model = torch.compile(model, backend=backend)

        sample = torch.randn(1, 3, 32, 32)
        output = compiled_model(sample)
        print(f"Model output shape: {output.shape}")

        df_logger.finalize()

        # Validate log files using the common utility
        expected_count = test_config.get("expected_events", 0)
        mode = test_config.get("mode", "exact")
        validate_log_files(log_file, test_name, expected_count, mode=mode)
    finally:
        # Cleanup test directories
        shutil.rmtree(test_base_dir, ignore_errors=True)

    return True


class TestDynamo:
    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    def test_dynamo_pytorch_model_tracing(self):
        test_config = {
            "name": "dynamo_pytorch_model_tracing",
            "expected_count": 10,
            "mode": "at_least",
            "env": {
                "DFTRACER_ENABLE": "1",
                "DFTRACER_TRACE_COMPRESSION": "0",
                "DFTRACER_INC_METADATA": "1",
                "DFTRACER_DISABLE_IO": "1",
            },
        }

        try:
            run_test_in_spawn_process(run_single_dynamo_test, test_config)
        except ImportError:
            pytest.skip("C++ dftracer library not available")

    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    def test_dynamo_disabled(self):
        test_config = {
            "name": "dynamo_disabled",
            "expected_count": 0,
            "mode": "exact",
            "env": {
                "DFTRACER_ENABLE": "0",
            },
        }

        try:
            run_test_in_spawn_process(run_single_dynamo_test, test_config)
        except ImportError:
            pytest.skip("C++ dftracer library not available")

    def test_dynamo_requires_pytorch(self):
        """Test that this test suite requires PyTorch to be available."""
        if not _check_torch_available():
            pytest.skip("PyTorch not available")
        else:
            print("PyTorch is available and ready for dynamo testing")

    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    def test_create_backend_with_autograd(self):
        """Test create_backend function with autograd enabled"""
        test_config = {
            "name": "create_backend_autograd",
            "autograd": True,
            "expected_count": 10,
            "mode": "at_least",
            "env": {
                "DFTRACER_ENABLE": "1",
                "DFTRACER_TRACE_COMPRESSION": "0",
                "DFTRACER_INC_METADATA": "1",
                "DFTRACER_DISABLE_IO": "1",
            },
        }

        try:
            run_test_in_spawn_process(run_create_backend_test, test_config)
        except ImportError:
            pytest.skip("C++ dftracer library not available")

    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    def test_create_backend_without_autograd(self):
        """Test create_backend function with autograd disabled"""
        test_config = {
            "name": "create_backend_no_autograd",
            "autograd": False,
            "expected_count": 10,
            "mode": "at_least",
            "env": {
                "DFTRACER_ENABLE": "1",
                "DFTRACER_TRACE_COMPRESSION": "0",
                "DFTRACER_INC_METADATA": "1",
                "DFTRACER_DISABLE_IO": "1",
            },
        }

        try:
            run_test_in_spawn_process(run_create_backend_test, test_config)
        except ImportError:
            pytest.skip("C++ dftracer library not available")

    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    def test_create_backend_disabled(self):
        """Test create_backend function when tracing is disabled"""
        test_config = {
            "name": "create_backend_disabled",
            "autograd": True,
            "expected_count": 0,
            "mode": "exact",
            "env": {
                "DFTRACER_ENABLE": "0",
            },
        }

        try:
            run_test_in_spawn_process(run_create_backend_test, test_config)
        except ImportError:
            pytest.skip("C++ dftracer library not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
