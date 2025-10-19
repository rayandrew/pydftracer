import os
import tempfile
from pathlib import Path

import pytest

from .utils import run_test_in_spawn_process


def _check_torch_available():
    """Check if torch is available in the current environment."""
    try:
        import torch  # noqa: F401 # type: ignore

        return True
    except ImportError:
        return False


def run_single_dynamo_test(test_config):
    import torch  # noqa: F401 # type: ignore
    from dftracer.python.dynamo import DFTRACER_ENABLE, dft_fn, dftracer

    try:
        import dftracer.dftracer  # noqa: F401
    except ImportError:
        print("WARNING: dftracer C++ library is NOT available - skipping test")
        return True

    log_file = test_config["log_file"]
    df_logger = dftracer.initialize_log(log_file, None, -1)
    dyn = dft_fn(name="dynamo", enabled=True)

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, 1)
            self.fc = torch.nn.Linear(16 * 15 * 15, 10)

        @dyn.compile
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

    if DFTRACER_ENABLE:
        if os.path.exists(log_file):
            with open(log_file, "rb") as f:
                data = f.read()
                assert b'"cat":"dynamo"' in data, (
                    "Expected dynamo category not found in log"
                )
                print("SUCCESS: DFTracer dynamo tracing is working")
        else:
            raise AssertionError("Log file was not created")

    return True


class TestDynamo:
    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    def test_dynamo_pytorch_model_tracing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = str(Path(temp_dir) / "dynamo.pfw")
            test_config = {
                "log_file": log_file,
                "env": {
                    "DFTRACER_ENABLE": "1",
                },
            }

            try:
                run_test_in_spawn_process(run_single_dynamo_test, test_config)
            except ImportError:
                pytest.skip("C++ dftracer library not available")

    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    def test_dynamo_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = str(Path(temp_dir) / "dynamo_disabled.pfw")
            test_config = {
                "log_file": log_file,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
