import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _check_torch_available():
    """Check if torch is available in the current environment."""
    try:
        import torch  # noqa: F401 # type: ignore

        return True
    except ImportError:
        return False


def run_single_dynamo_test(log_file):
    import torch  # noqa: F401 # type: ignore

    from dftracer.dynamo import dft_fn, dftracer

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

    try:
        model = SimpleModel()
        t_model: torch.nn.Module = model  # type: ignore
        # Create random input
        sample = torch.randn(1, 3, 32, 32)
        output = t_model(sample)
        print(f"Model output shape: {output.shape}")
        df_logger.finalize()

        # Verify the log file contains expected tracing data
        if os.path.exists(log_file):
            with open(log_file, "rb") as f:
                data = f.read()
                if b'"cat":"dynamo"' in data:
                    print("SUCCESS: DFTracer dynamo tracing is working")
                    sys.exit(0)
                else:
                    print("ERROR: Expected dynamo category not found in log")
                    sys.exit(1)
        else:
            print("ERROR: Log file was not created")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


class TestDynamo:
    @pytest.mark.subprocess
    @pytest.mark.skipif(not _check_torch_available(), reason="PyTorch not available")
    def test_dynamo_pytorch_model_tracing(self):
        """Test that dynamo can trace PyTorch models correctly using subprocess execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "dynamo.pfw"

            test_script = f'''
import sys
import os
sys.path.insert(0, "{os.path.dirname(os.path.dirname(__file__))}")

os.environ["DFTRACER_ENABLE"] = "1"

from tests.test_dynamo import run_single_dynamo_test

if __name__ == "__main__":
    try:
        result = run_single_dynamo_test("{log_file}")
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test failed with error: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit
'''

            # Write the test script to a temporary file
            script_file = Path(temp_dir) / "dynamo_test_script.py"
            script_file.write_text(test_script)

            env = os.environ.copy()
            env["DFTRACER_ENABLE"] = "1"
            env["PYTHONPATH"] = str(Path(__file__).parent.parent)

            result = subprocess.run(
                [sys.executable, str(script_file)],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Check results
            print(f"Return code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")

            # Assertions
            assert result.returncode == 0, (
                f"Test script failed with return code {result.returncode}"
            )
            assert "SUCCESS: DFTracer dynamo tracing is working" in result.stdout
            assert log_file.exists(), "Log file was not created"

            # Verify log file contents
            with open(log_file, "rb") as f:
                log_data = f.read()
                assert b'"cat":"dynamo"' in log_data, (
                    "Expected dynamo category not found in log file"
                )

    def test_dynamo_requires_pytorch(self):
        """Test that this test suite requires PyTorch to be available."""
        if not _check_torch_available():
            pytest.skip("PyTorch not available")
        else:
            print("PyTorch is available and ready for dynamo testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
