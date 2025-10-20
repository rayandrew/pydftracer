Examples
========

This section provides practical examples of using pydftracer in different scenarios.

.. contents:: Table of Contents
   :local:
   :depth: 2

Python Examples
---------------

Application Level Example
~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates using pydftracer at the application level with explicit
initialization and function decorators.

.. code-block:: python
   :linenos:

    from dftracer.python import dftracer, dft_fn
    import numpy as np
    import os
    from time import sleep
    from multiprocessing import get_context

    # Initialize DFTracer
    log_inst = dftracer.initialize_log(logfile=None, data_dir=None, process_id=-1)

    # Create function tracer for compute operations
    compute_tracer = dft_fn("COMPUTE")

    # Example of using function decorators
    @compute_tracer.log
    def log_events(index):
        sleep(1)

    # Example of function spawning and implicit I/O calls
    def posix_calls(val):
        index, is_spawn = val
        cwd = os.getcwd()
        path = f"{cwd}/data/demofile{index}.txt"
        f = open(path, "w+")
        f.write("Now the file has more content!")
        f.close()

        if is_spawn:
            print(f"Calling spawn on {index} with pid {os.getpid()}")
            log_inst.finalize()  # Finalize DFTracer in spawned process
        else:
            print(f"Not calling spawn on {index} with pid {os.getpid()}")

    # NPZ operations (internally calls POSIX)
    def npz_calls(index):
        cwd = os.getcwd()
        path = f"{cwd}/data/demofile{index}.npz"
        if os.path.exists(path):
            os.remove(path)
        records = np.random.randint(255, size=(8, 8, 1024), dtype=np.uint8)
        record_labels = [0] * 1024
        np.savez(path, x=records, y=record_labels)

    def main():
        log_events(0)
        npz_calls(1)

        # Spawn processes for parallel operations
        with get_context('spawn').Pool(1) as pool:
            pool.map(posix_calls, ((2, True),))

        log_inst.finalize()

    if __name__ == "__main__":
        main()

**Environment Configuration:**

For this example, you need to set the following environment variables:

.. code-block:: bash
   :linenos:

    # Log file path (process ID, app name, and .pfw will be appended)
    # Final log file: ~/log_file-<APP_NAME>-<PID>.pfw
    export DFTRACER_LOG_FILE=~/log_file

    # Colon-separated paths for profiling
    export DFTRACER_DATA_DIR=/dev/shm/:/p/gpfs1/$USER/dataset:$PWD/data

    # Enable DFTracer
    export DFTRACER_ENABLE=1

LD_PRELOAD Example
~~~~~~~~~~~~~~~~~~

This example shows using DFTracer with LD_PRELOAD for automatic I/O interception
without explicit initialization in the code.

.. code-block:: python
   :linenos:

    import numpy as np
    import os
    from multiprocessing import get_context

    # Example of function spawning and implicit I/O calls
    def posix_calls(val):
        index, is_spawn = val
        cwd = os.getcwd()
        path = f"{cwd}/data/demofile{index}.txt"
        f = open(path, "w+")
        f.write("Now the file has more content!")
        f.close()

        if is_spawn:
            print(f"Calling spawn on {index} with pid {os.getpid()}")
        else:
            print(f"Not calling spawn on {index} with pid {os.getpid()}")

    # NPZ operations (internally calls POSIX)
    def npz_calls(index):
        cwd = os.getcwd()
        path = f"{cwd}/data/demofile{index}.npz"
        if os.path.exists(path):
            os.remove(path)
        records = np.random.randint(255, size=(8, 8, 1024), dtype=np.uint8)
        record_labels = [0] * 1024
        np.savez(path, x=records, y=record_labels)

    def main():
        npz_calls(1)

        with get_context('spawn').Pool(1) as pool:
            pool.map(posix_calls, ((2, True),))

    if __name__ == "__main__":
        main()

**Environment Configuration:**

.. code-block:: bash
   :linenos:

    # Log file path (process ID, app name, and .pfw will be appended)
    export DFTRACER_LOG_FILE=~/log_file

    # Colon-separated paths for profiling
    export DFTRACER_DATA_DIR=/dev/shm/:/p/gpfs1/$USER/dataset

    # Set initialization mode to PRELOAD
    export DFTRACER_INIT=PRELOAD

    # Enable DFTracer
    export DFTRACER_ENABLE=1

    # Run with LD_PRELOAD
    LD_PRELOAD=/path/to/libdftracer_preload.so python your_script.py

.. _python-hybrid-mode:

Hybrid Mode Example
~~~~~~~~~~~~~~~~~~~

This example demonstrates the hybrid mode, combining both application-level
initialization and LD_PRELOAD for comprehensive profiling.

.. code-block:: python
   :linenos:

    from dftracer.python import dftracer, dft_fn
    import numpy as np
    import os
    from time import sleep
    from multiprocessing import get_context

    # Initialize DFTracer at application level
    log_inst = dftracer.initialize_log(logfile=None, data_dir=None, process_id=-1)
    compute_tracer = dft_fn("COMPUTE")

    # Example of using function decorators
    @compute_tracer.log
    def log_events(index):
        sleep(1)

    # Example of function spawning and implicit I/O calls
    def posix_calls(val):
        index, is_spawn = val
        cwd = os.getcwd()
        path = f"{cwd}/data/demofile{index}.txt"
        f = open(path, "w+")
        f.write("Now the file has more content!")
        f.close()

        if is_spawn:
            print(f"Calling spawn on {index} with pid {os.getpid()}")
            log_inst.finalize()
        else:
            print(f"Not calling spawn on {index} with pid {os.getpid()}")

    # NPZ operations
    def npz_calls(index):
        cwd = os.getcwd()
        path = f"{cwd}/data/demofile{index}.npz"
        if os.path.exists(path):
            os.remove(path)
        records = np.random.randint(255, size=(8, 8, 1024), dtype=np.uint8)
        record_labels = [0] * 1024
        np.savez(path, x=records, y=record_labels)

    def main():
        log_events(0)
        npz_calls(1)

        with get_context('spawn').Pool(1) as pool:
            pool.map(posix_calls, ((2, True),))

        log_inst.finalize()

    if __name__ == "__main__":
        main()

**Environment Configuration:**

.. code-block:: bash
   :linenos:

    # Log file path
    export DFTRACER_LOG_FILE=~/log_file

    # Data directories to profile
    export DFTRACER_DATA_DIR=/dev/shm/:/p/gpfs1/$USER/dataset

    # Set to PRELOAD mode
    export DFTRACER_INIT=PRELOAD

    # Enable DFTracer
    export DFTRACER_ENABLE=1

    # Run with LD_PRELOAD
    LD_PRELOAD=/path/to/libdftracer_preload.so python your_script.py

Deep Learning Example: ResNet50 on ALCF Polaris
------------------------------------------------

This example shows how to profile a ResNet50 training workload using PyTorch
and torchvision on the Polaris supercomputer at Argonne Leadership Computing Facility.

Environment Setup
~~~~~~~~~~~~~~~~~

Create a conda environment and install dependencies:

.. code-block:: bash
   :linenos:

    #!/bin/bash +x
    set -e
    set -x

    export MODULEPATH=/soft/modulefiles/conda/:$MODULEPATH
    module load 2023-10-04  # Latest conda module on Polaris

    export ML_ENV=$PWD/PolarisAT/conda-envs/ml_workload_latest_conda_2

    if [[ -e $ML_ENV ]]; then
        conda activate $ML_ENV
    else
        # Clone base environment
        conda create -p $ML_ENV --clone /soft/datascience/conda/2023-10-04/mconda3/
        conda activate $ML_ENV

        # Install MPI4Py with GPU support
        yes | MPICC="cc -shared -target-accel=nvidia80" \
            pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

        # Install pydftracer
        yes | pip install --no-cache-dir git+https://github.com/hariharan-devarajan/dftracer.git

        # Reinstall torch and horovod
        pip uninstall -y torch horovod
        yes | pip install --no-cache-dir horovod
    fi

Application Instrumentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since ``torchvision.datasets.ImageFolder`` spawns separate Python processes for
parallel data loading, we use **hybrid mode** (see :ref:`python-hybrid-mode`) to
capture I/O from both the main process and spawned workers.

.. code-block:: python
   :linenos:

    import os
    from dftracer.python import dftracer as logger, dft_fn as dft_event_logging

    # Initialize DFTracer
    dft_pid = os.getpid()
    log_inst = logger.initialize_log(
        f"./resnet50/dft_fn_py_level-{dft_pid}.pfw",
        "",
        dft_pid
    )

    # Create tracers for different operation types
    compute_dft = dft_event_logging("Compute")
    io_dft = dft_event_logging("IO", name="real_IO")

    def train(epoch, model, train_loader, criterion, device):
        """Training loop with DFTracer instrumentation"""

        # Trace data loading iterations
        for i, (images, target) in io_dft.iter(enumerate(train_loader)):

            # Trace CPU to GPU transfer
            with dft_event_logging(
                "communication-except-io",
                name="cpu-gpu-transfer",
                step=i,
                epoch=epoch
            ) as transfer:
                images = images.to(device)
                target = target.to(device)

            # Trace forward propagation
            with dft_event_logging(
                "compute",
                name="model-compute-forward-prop",
                step=i,
                epoch=epoch
            ) as compute:
                output = model(images)
                loss = criterion(output, target)

            # Trace backward propagation
            with dft_event_logging(
                "compute",
                name="model-compute-backward-prop",
                step=i,
                epoch=epoch
            ) as compute:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

    def main():
        # ... model setup and training ...

        # Finalize DFTracer
        log_inst.finalize()

    if __name__ == "__main__":
        main()

Job Submission Script
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash
   :linenos:

    #!/bin/bash

    # Load environment
    export MODULEPATH=/soft/modulefiles/conda/:$MODULEPATH
    module load 2023-10-04
    conda activate ./dlio_ml_workloads/PolarisAT/conda-envs/ml_workload_latest_conda

    # Set library path
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

    # DFTracer configuration
    export DFTRACER_LOG_LEVEL=ERROR
    export DFTRACER_ENABLE=1
    export DFTRACER_INC_METADATA=1
    export DFTRACER_INIT=PRELOAD

    # Path to ResNet50 dataset
    export DFTRACER_DATA_DIR=./resnet_original_data

    # POSIX-level log file
    export DFTRACER_LOG_FILE=./dft_fn_posix_level.pfw

    # Run with LD_PRELOAD
    LD_PRELOAD=$CONDA_PREFIX/lib/python*/site-packages/dftracer/lib/libdftracer_preload.so \
        aprun -n 4 -N 4 python resnet_hvd_dlio.py \
        --batch-size 64 \
        --epochs 1 \
        > dft_fn.log 2>&1

    # Combine all trace files
    cat *.pfw > combined_logs.pfw

Understanding the Output
~~~~~~~~~~~~~~~~~~~~~~~~

This configuration produces two types of trace files:

1. **Python-level traces** (``dft_fn_py_level-*.pfw``): Function-level events from decorators
2. **POSIX-level traces** (``dft_fn_posix_level-*.pfw``): Low-level I/O operations

Combine them using:

.. code-block:: bash

    cat *.pfw > combined_logs.pfw

Integrated Applications
-----------------------

pydftracer is currently used in production by several applications:

1. **DLIO Benchmark** - `GitHub <https://github.com/argonne-lcf/dlio_benchmark>`_

   Comprehensive I/O benchmark for deep learning workloads

2. **MuMMI** - Multiscale Machine-learned Modeling Infrastructure

   Large-scale molecular dynamics simulations

3. **ResNet50 Training** - PyTorch and torchvision

   Image classification with distributed training

Additional Resources
--------------------

For more examples and use cases, see:

- :doc:`quickstart` - Basic usage patterns
- :doc:`ai_ml_guide` - AI/ML specific features
- :doc:`dynamo_guide` - PyTorch Dynamo integration
