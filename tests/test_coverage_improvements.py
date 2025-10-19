#!/usr/bin/env python
"""
Test suite to improve coverage for specific uncovered lines
"""

import importlib
import os
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from dftracer.python import compute, dft_fn, dftracer


def run_ai_update_method_parameter_paths_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test_ai_coverage.pfw")

        # Initialize dftracer
        df_logger = dftracer.initialize_log(log_file, None, -1)

        # Test case 1: Test with epoch parameter
        compute.update(epoch=5)
        assert "epoch" in compute.profiler._arguments_int
        assert (
            compute.profiler._arguments_int["epoch"][1] == 5
        )  # TagValueTuple: (tag_type, value)

        # Test case 2: Test with step parameter
        compute.update(step=100)
        assert "epoch" in compute.profiler._arguments_int
        assert compute.profiler._arguments_int["epoch"][1] == 5
        assert "step" in compute.profiler._arguments_int
        assert compute.profiler._arguments_int["step"][1] == 100

        # Test case 3: Test with image_idx parameter
        compute.update(image_idx=42)
        assert compute.profiler._arguments_int["epoch"][1] == 5
        assert compute.profiler._arguments_int["step"][1] == 100
        assert compute.profiler._arguments_int["image_idx"][1] == 42

        # Test case 4: Test with image_size parameter
        compute.update(image_size=224)
        assert compute.profiler._arguments_int["epoch"][1] == 5
        assert compute.profiler._arguments_int["step"][1] == 100
        assert compute.profiler._arguments_int["image_idx"][1] == 42
        assert compute.profiler._arguments_float["image_size"][1] == 224.0

        # Test case 5: Test with custom args
        custom_args = {"learning_rate": 0.001, "batch_size": 16}
        compute.update(args=custom_args)
        assert compute.profiler._arguments_int["epoch"][1] == 5
        assert compute.profiler._arguments_int["step"][1] == 100
        assert compute.profiler._arguments_int["image_idx"][1] == 42
        assert compute.profiler._arguments_float["image_size"][1] == 224.0
        assert compute.profiler._arguments_float["learning_rate"][1] == 0.001
        assert compute.profiler._arguments_int["batch_size"][1] == 16

        # Test case 6: Test with all parameters together
        compute.update(
            epoch=10,
            step=200,
            image_idx=25,
            image_size=512,
            args={"optimizer": "adam", "loss": 0.5},
        )
        assert compute.profiler._arguments_int["epoch"][1] == 10
        assert compute.profiler._arguments_int["step"][1] == 200
        assert compute.profiler._arguments_int["image_idx"][1] == 25
        assert compute.profiler._arguments_float["image_size"][1] == 512.0
        assert compute.profiler._arguments_string["optimizer"][1] == "adam"
        assert compute.profiler._arguments_float["loss"][1] == 0.5

        # Test case 7: Test with None values (should not add to arguments)
        compute.update(epoch=None, step=None, image_idx=None, image_size=None, args={})
        # Values should remain from test case 6
        assert compute.profiler._arguments_int["epoch"][1] == 10
        assert compute.profiler._arguments_int["step"][1] == 200
        assert compute.profiler._arguments_int["image_idx"][1] == 25
        assert compute.profiler._arguments_float["image_size"][1] == 512.0
        assert compute.profiler._arguments_string["optimizer"][1] == "adam"
        assert compute.profiler._arguments_float["loss"][1] == 0.5

        # Test case 8: Specifically target the missing lines with individual calls
        compute.update(step=999)
        compute.update(image_idx=123)
        compute.update(image_size=128 * 128)
        assert compute.profiler._arguments_int["step"][1] == 999
        assert compute.profiler._arguments_int["image_idx"][1] == 123
        assert compute.profiler._arguments_float["image_size"][1] == 16384.0

        df_logger.finalize()


def run_dft_fn_log_decorator_attribute_access_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test_decorator_attrs.pfw")
        df_logger = dftracer.initialize_log(log_file, None, -1)

        # Create a test class with attributes that the decorator looks for
        class TestObject:
            def __init__(self):
                self.epoch = 5
                self.step = 100
                self.image_size = 224
                self.image_idx = 42

            def process_method(self, x):
                return x * 2

        tracer = dft_fn("test_attr_access", enable=True)

        # Test decorator that accesses object attributes
        obj = TestObject()
        decorated_method = tracer.log(obj.process_method)

        # This should trigger the attribute access lines in the decorator
        result = decorated_method(5)
        assert result == 10

        df_logger.finalize()


def run_dft_fn_log_init_with_constructor_args_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test_log_init.pfw")
        df_logger = dftracer.initialize_log(log_file, None, -1)

        tracer = dft_fn("test_log_init", enable=True)

        # Test class with constructor that has the special arguments
        class TestClass:
            @tracer.log_init
            def __init__(self, epoch=None, step=None, image_idx=None, image_size=None):
                self.epoch = epoch
                self.step = step
                self.image_idx = image_idx
                self.image_size = image_size

        obj1 = TestClass(epoch=10)
        obj2 = TestClass(image_idx=5)
        obj3 = TestClass(image_size=64)
        obj4 = TestClass(step=25)

        # Test with multiple arguments
        obj5 = TestClass(epoch=1, step=2, image_idx=3, image_size=32)

        assert obj1 is not None
        assert obj2 is not None
        assert obj3 is not None
        assert obj4 is not None
        assert obj5 is not None
        assert obj1.epoch == 10

        df_logger.finalize()


def run_iter_method_coverage_test():
    """Test the iter method in logger.py for coverage."""
    import dftracer.python.logger  # noqa: F401

    importlib.reload(sys.modules["dftracer.python.logger"])

    from dftracer.python import dft_fn, dftracer

    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test_iter.pfw")
        df_logger = dftracer.initialize_log(log_file, None, -1)

        tracer = dft_fn("test_iter", enable=True)

        # Test iter method with different sequence types
        test_data = [1, 2, 3, 4, 5]

        # This should cover the iter method implementation
        result = list(tracer.iter(test_data))
        assert result == test_data

        # Test with disabled tracer
        tracer_disabled = dft_fn("test_iter_disabled", enable=False)
        result_disabled = list(tracer_disabled.iter(test_data))
        assert result_disabled == test_data

        df_logger.finalize()


def run_log_from_function_params_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        logfile = os.path.join(temp_dir, "test_log_from_function_params.pfw")
        df_logger = dftracer.initialize_log(logfile, None, -1)
        profiler = dft_fn("test_log_from_function_params")

        @profiler.log
        def fun(epoch, step, image_size, image_idx):
            return 0

        assert fun(epoch=1, step=2, image_size=3, image_idx=4) == 0
        assert profiler._arguments_int["epoch"][1] == 1
        assert profiler._arguments_int["step"][1] == 2
        assert profiler._arguments_float["image_size"][1] == 3.0
        assert profiler._arguments_int["image_idx"][1] == 4

        df_logger.finalize()


def run_log_from_method_params_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test_log_from_method_params.pfw")
        df_logger = dftracer.initialize_log(log_file, None, -1)
        profiler = dft_fn("test_log_from_method_params")

        class F:
            def __init__(self, epoch, step, image_size, image_idx) -> None:
                self.epoch = epoch
                self.step = step
                self.image_size = image_size
                self.image_idx = image_idx

            @profiler.log
            def fun(self):
                return 0

        f = F(epoch=1, step=2, image_size=3, image_idx=4)
        assert f.fun() == 0
        assert profiler._arguments_int["epoch"][1] == 1
        assert profiler._arguments_int["step"][1] == 2
        assert profiler._arguments_float["image_size"][1] == 3.0
        assert profiler._arguments_int["image_idx"][1] == 4

        df_logger.finalize()


def run_log_init_from_constructor_params_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test_log_init_from_constructor_params.pfw")
        df_logger = dftracer.initialize_log(log_file, None, -1)
        profiler = dft_fn("test_log_init_from_constructor_params")

        class F:
            @profiler.log_init
            def __init__(self, epoch, step, image_size, image_idx) -> None:
                pass

        F(epoch=1, step=2, image_size=3, image_idx=4)
        assert profiler._arguments_int["epoch"][1] == 1
        assert profiler._arguments_int["step"][1] == 2
        assert profiler._arguments_float["image_size"][1] == 3.0
        assert profiler._arguments_int["image_idx"][1] == 4

        df_logger.finalize()


class TestCoverageImprovements:
    """Test class to improve coverage for missing lines."""

    def run_test_in_subprocess(self, test_function_name):
        """Helper method to run a test function in subprocess with proper environment."""
        script_content = f'''
import sys
import os
sys.path.insert(0, "{os.path.dirname(os.path.dirname(__file__))}")

# Set environment variable
os.environ["DFTRACER_ENABLE"] = "1"

from tests.test_coverage_improvements import {test_function_name}

if __name__ == "__main__":
    try:
        {test_function_name}()
        print("Test completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"Test failed with error: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            print("Test subprocess output:")
            print(result.stdout)
            if result.stderr:
                print("Test subprocess errors:")
                print(result.stderr)

            assert result.returncode == 0, (
                f"Test {test_function_name} failed in subprocess with return code {result.returncode}"
            )

        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)

    def test_ai_update_method_parameter_paths(self):
        """Test AI update method to cover all parameter assignment paths."""
        self.run_test_in_subprocess("run_ai_update_method_parameter_paths_test")

    def test_dft_fn_log_decorator_attribute_access(self):
        """Test log decorator with attribute access to cover missing lines in logger.py."""
        self.run_test_in_subprocess("run_dft_fn_log_decorator_attribute_access_test")

    def test_dft_fn_log_init_with_constructor_args(self):
        """Test log_init decorator with constructor arguments."""
        self.run_test_in_subprocess("run_dft_fn_log_init_with_constructor_args_test")

    def test_iter_method_coverage(self):
        """Test the iter method in logger.py for coverage."""
        self.run_test_in_subprocess("run_iter_method_coverage_test")

    def test_log_from_function_params(self):
        """Test logging from function parameters."""
        self.run_test_in_subprocess("run_log_from_function_params_test")

    def test_log_from_method_params(self):
        """Test logging from method parameters."""
        self.run_test_in_subprocess("run_log_from_method_params_test")

    def test_log_init_from_constructor_params(self):
        """Test logging from constructor parameters."""
        self.run_test_in_subprocess("run_log_init_from_constructor_params_test")

    def test_noop_profiler_coverage(self):
        """Test NoOpProfiler to cover fallback cases."""
        # This test doesn't need subprocess since it doesn't use dftracer.initialize_log
        # Mock import error to trigger NoOpProfiler usage
        with patch("dftracer.python.common.profiler", side_effect=ImportError()):
            import dftracer.python.logger  # noqa: F401

            importlib.reload(sys.modules["dftracer.python.logger"])

            from dftracer.python import NoOpProfiler

            noop = NoOpProfiler()

            # Test all NoOpProfiler methods to ensure coverage
            noop.initialize("test.log", "/tmp", 123)
            assert noop.get_time() == 0
            noop.enter_event()
            noop.exit_event()
            noop.log_event(
                name="test",
                cat="test_cat",
                start_time=100,
                duration=50,
                int_args={"test_int": 1},
                string_args={"test_str": "value"},
                float_args={"test_float": 1.5},
            )
            noop.log_metadata_event("test_key", "test_value")
            noop.finalize()

    def test_profiler_with_disabled_dftracer(self):
        """Test profiler behavior when DFTRACER_ENABLE is False."""
        # This test doesn't need subprocess since it tests disabled behavior
        with patch.dict(os.environ, {"DFTRACER_ENABLE": "0"}):
            import dftracer.python.logger  # noqa: F401

            importlib.reload(sys.modules["dftracer.python.logger"])

            from dftracer.python import compute, dftracer

            # Test that operations work even when disabled
            with tempfile.TemporaryDirectory() as temp_dir:
                log_file = os.path.join(temp_dir, "test_disabled.pfw")
                df_logger = dftracer.initialize_log(log_file, None, -1)

                # These should work even when disabled
                compute.update(epoch=1, step=10)

                df_logger.finalize()

    def test_signal_handlers_coverage(self):
        """Test signal handler registration for coverage."""

        # Test that signal handlers are properly registered
        # We can't easily test the actual signal handling without triggering it,
        # but we can verify the setup code path is covered
        from dftracer.python.common import capture_signal

        # Test the capture_signal function with a mock
        mock_frame = MagicMock()

        with patch("dftracer.python.common.dftracer") as mock_dftracer:
            mock_instance = MagicMock()
            mock_dftracer.get_instance.return_value = mock_instance

            with patch("sys.exit") as mock_exit:
                capture_signal(2, mock_frame)  # SIGINT
                mock_instance.finalize.assert_called_once()
                mock_exit.assert_called_once_with(2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
