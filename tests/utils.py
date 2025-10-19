import os
import sys
from multiprocessing import Manager, get_context


def _worker_process(func_name, module_name, config, queue):
    """Worker function to run in spawned process - must be at module level for pickling"""
    import traceback

    # Don't capture stdout/stderr - let C++ errors go directly to console
    # This allows us to see SIGABRT messages from the C++ library

    try:
        print(f"Worker process starting - Python: {sys.version}", file=sys.stderr)
        print(f"Worker CWD: {os.getcwd()}", file=sys.stderr)
        print(
            f"DFTRACER_ENABLE: {os.environ.get('DFTRACER_ENABLE', 'NOT SET')}",
            file=sys.stderr,
        )

        import importlib

        module = importlib.import_module(module_name)
        test_func = getattr(module, func_name)

        print(f"Successfully imported {module_name}.{func_name}", file=sys.stderr)
        sys.stderr.flush()

        result = test_func(config)

        print("Test completed successfully", file=sys.stderr)
        sys.stderr.flush()

        queue.put(
            {
                "success": result,
                "stdout": "",
                "stderr": "",
                "error": None,
            }
        )
    except Exception as e:
        sys.stderr.flush()

        queue.put(
            {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
    except BaseException as e:
        sys.stderr.flush()

        queue.put(
            {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"BaseException: {str(e)}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def run_test_in_spawn_process(test_func, test_config, timeout=120):
    """
    Run a test function in a separate process using multiprocessing spawn.

    Args:
        test_func: The test function to run (must be a module-level function)
        test_config: Configuration dictionary to pass to the test function
        timeout: Timeout in seconds (default: 120)

    Returns:
        None (raises AssertionError if test fails)

    Raises:
        TimeoutError: If the test times out
        RuntimeError: If the process fails to return results
        AssertionError: If the test fails
    """

    # Use spawn context to ensure fresh process
    ctx = get_context("spawn")

    # Create a managed queue to ensure proper sharing across spawned processes
    manager = Manager()
    output_queue = manager.Queue()

    func_name = test_func.__name__
    module_name = test_func.__module__

    # Extract environment variables from config
    env_vars = test_config.get("env", {})

    # Temporarily modify parent environment before spawning
    # With spawn, child inherits parent's environment at spawn time
    old_env = {}
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Create and start process - it will inherit the modified environment
        process = ctx.Process(
            target=_worker_process,
            args=(func_name, module_name, test_config, output_queue),
        )
        process.start()

        # Wait for completion with timeout
        process.join(timeout=timeout)
    finally:
        # Restore original environment in parent
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    # Check if process is still alive (timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        raise TimeoutError(
            f"Test {test_config.get('name', 'unknown')} timed out after {timeout} seconds"
        )

    # Check exit code first
    if process.exitcode != 0:
        # Try to get error info from queue if available
        error_msg = f"Test {test_config.get('name', 'unknown')} process exited with code {process.exitcode}"
        try:
            if not output_queue.empty():
                result_data = output_queue.get(timeout=1)
                if result_data.get("error"):
                    error_msg += f"\nError: {result_data['error']}"
                if result_data.get("traceback"):
                    error_msg += f"\nTraceback:\n{result_data['traceback']}"
                if result_data.get("stderr"):
                    error_msg += f"\nStderr:\n{result_data['stderr']}"
        except Exception:
            # Ignore errors when trying to get additional error info
            pass
        raise RuntimeError(error_msg)

    # Get results from queue
    try:
        result_data = output_queue.get(timeout=5)
    except Exception as e:
        raise RuntimeError(
            f"Test {test_config.get('name', 'unknown')} process completed but failed to return results. "
            f"Exit code: {process.exitcode}"
        ) from e

    print(f"Test {test_config.get('name', 'unknown')} process output:")
    print(result_data["stdout"])
    if result_data["stderr"]:
        print(f"Test {test_config.get('name', 'unknown')} process errors:")
        print(result_data["stderr"])

    if result_data["error"]:
        print(f"Test failed with error: {result_data['error']}")
        if "traceback" in result_data:
            print(result_data["traceback"])
        raise AssertionError(
            f"Test {test_config.get('name', 'unknown')} failed in spawned process: {result_data['error']}"
        )

    assert result_data["success"], (
        f"Test {test_config.get('name', 'unknown')} failed in spawned process"
    )


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
            dftracer_libs_path = os.path.join(site_dir, "dftracer")

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
