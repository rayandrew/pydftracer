PyTorch Dynamo Integration
===========================

This guide explains how to use pydftracer with PyTorch's Dynamo compiler infrastructure
to trace and profile PyTorch models and operations.

Overview
--------

The Dynamo integration allows you to:

- Profile PyTorch models compiled with ``torch.compile()``
- Trace individual operations in the computation graph
- Track forward and backward passes
- Monitor gradient computation
- Analyze model performance layer by layer

Prerequisites
-------------

Install PyTorch and the dynamo optional dependencies:

.. code-block:: bash

   pip install pydftracer[dynamo]

Or manually:

.. code-block:: bash

   pip install torch>=2.5.1

Setup
-----

Enable DFTracer before using Dynamo integration:

.. code-block:: bash

   export DFTRACER_ENABLE=1
   export DFTRACER_DISABLE_IO=1  # Optional: disable I/O tracing for pure compute profiling

Using the Dynamo Decorator
---------------------------

Basic Model Tracing
~~~~~~~~~~~~~~~~~~~

The simplest way to use Dynamo integration is with the ``@dynamo.compile`` decorator:

.. code-block:: python

   import torch
   from dftracer.python import dftracer, dynamo

   # Initialize logger
   df_logger = dftracer.initialize_log("dynamo_trace.pfw", "/tmp/data", -1)

   # Define a model
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

   # Create model and run
   model = SimpleModel()
   sample = torch.randn(1, 3, 32, 32)
   output = model(sample)

   # Finalize
   df_logger.finalize()

Using torch.compile Backend
----------------------------

Alternative Backend Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use DFTracer as a custom backend for ``torch.compile()``:

.. code-block:: python

   import torch
   from dftracer.python import dftracer, dftracer_dynamo_backend

   # Initialize logger
   df_logger = dftracer.initialize_log("dynamo_trace.pfw", "/tmp/data", -1)

   # Define your model
   class MyModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = torch.nn.Sequential(
               torch.nn.Linear(784, 256),
               torch.nn.ReLU(),
               torch.nn.Linear(256, 10)
           )

       def forward(self, x):
           return self.layers(x)

   # Compile with DFTracer backend
   model = MyModel()
   compiled_model = torch.compile(model, backend=dftracer_dynamo_backend)

   # Run inference
   input_tensor = torch.randn(32, 784)
   output = compiled_model(input_tensor)

   df_logger.finalize()

Advanced Configuration
----------------------

Custom Dynamo Tracer
~~~~~~~~~~~~~~~~~~~~

For more control, create a custom Dynamo tracer instance:

.. code-block:: python

   from dftracer.python import Dynamo

   # Create a custom Dynamo tracer
   dynamo_tracer = Dynamo(
       name="resnet50",
       epoch=1,
       step=100,
       image_idx=42,
       image_size=(224, 224),
       enable=True
   )

   # Use in your training loop
   # The tracer will automatically track operations

Training Loop Integration
-------------------------

Complete Training Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from dftracer.python import dftracer, dynamo

   # Initialize logger
   df_logger = dftracer.initialize_log("training.pfw", "/tmp/data", -1)

   # Model definition
   class ConvNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(1, 32, 3, 1)
           self.conv2 = nn.Conv2d(32, 64, 3, 1)
           self.fc1 = nn.Linear(9216, 128)
           self.fc2 = nn.Linear(128, 10)

       @dynamo.compile
       def forward(self, x):
           x = self.conv1(x)
           x = torch.nn.functional.relu(x)
           x = self.conv2(x)
           x = torch.nn.functional.relu(x)
           x = torch.nn.functional.max_pool2d(x, 2)
           x = torch.flatten(x, 1)
           x = self.fc1(x)
           x = torch.nn.functional.relu(x)
           x = self.fc2(x)
           return x

   # Training setup
   model = ConvNet()
   optimizer = torch.optim.Adam(model.parameters())
   criterion = nn.CrossEntropyLoss()

   # Training loop
   for epoch in range(5):
       for batch_idx, (data, target) in enumerate(train_loader):
           optimizer.zero_grad()

           # Forward pass (traced by Dynamo)
           output = model(data)

           # Compute loss and backward
           loss = criterion(output, target)
           loss.backward()

           # Optimizer step
           optimizer.step()

   df_logger.finalize()

What Gets Traced
----------------

Operation Details
~~~~~~~~~~~~~~~~~

The Dynamo integration traces:

- **Module calls**: Conv2d, Linear, BatchNorm, etc.
- **Function calls**: ReLU, MaxPool, matrix operations
- **Method calls**: Tensor operations
- **Metadata**: Operation names, types, targets
- **Timing**: Start and end timestamps
- **Gradient tracking**: Whether gradients are enabled

Traced Information
~~~~~~~~~~~~~~~~~~

For each operation, DFTracer records:

- Operation name (with layer/function info)
- Operation type (call_module, call_function, call_method)
- Target (module path, function name, method name)
- Timestamp (microseconds)
- Gradient enabled status
- Duration

Environment Variables
---------------------

Dynamo-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Enable DFTracer
   export DFTRACER_ENABLE=1

   # Disable I/O tracing (focus on compute)
   export DFTRACER_DISABLE_IO=1

   # Include metadata in traces
   export DFTRACER_INC_METADATA=1

   # Disable compression for easier debugging
   export DFTRACER_TRACE_COMPRESSION=0

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: Dynamo integration not working

**Solution**: Ensure you have PyTorch >= 2.5.1 and DFTRACER_ENABLE=1

.. code-block:: bash

   pip install torch>=2.5.1
   export DFTRACER_ENABLE=1

**Issue**: ImportError for functorch or dynamo

**Solution**: Install the dynamo optional dependencies

.. code-block:: bash

   pip install pydftracer[dynamo]

Performance Considerations
--------------------------

Overhead
~~~~~~~~

- The Dynamo integration adds minimal overhead
- Most overhead comes from the Dynamo compilation itself

Best Practices
~~~~~~~~~~~~~~

1. **Use selective tracing**: Only trace the operations you need
2. **Disable I/O tracing**: Set ``DFTRACER_DISABLE_IO=1`` for compute-only profiling
3. **Batch operations**: Profile on representative batch sizes
4. **Warm-up runs**: Run a few iterations before profiling to account for compilation

Example Output
--------------

Trace File Contents
~~~~~~~~~~~~~~~~~~~

After running with Dynamo tracing, your trace file will contain entries like:

.. code-block:: json

  [
      {"id":1,"name":"HH","cat":"dftracer","pid":3200613,"tid":3200613,"ph":"M","args":{"hhash":"f9ff883caaf21863","name":"tuolumne1004","value":"f9ff883caaf21863"}}
      {"id":2,"name":"thread_name","cat":"dftracer","pid":3200613,"tid":3200613,"ph":"M","args":{"hhash":"f9ff883caaf21863","name":"3200613","value":"thread_name"}}
      {"id":3,"name":"FH","cat":"dftracer","pid":3200613,"tid":3200613,"ph":"M","args":{"hhash":"f9ff883caaf21863","name":"/usr/WS2/sinurat1/pydftracer","value":"457466652c169c22"}}
      {"id":4,"name":"SH","cat":"dftracer","pid":3200613,"tid":3200613,"ph":"M","args":{"hhash":"f9ff883caaf21863","name":"/usr/WS2/sinurat1/pydftracer/.venv-tuo/bin/python;-c;from multiprocessing.spawn import spawn_main; spawn_main tracker_fd=8, pipe_handle=11 ;--multiprocessing-fork","value":"b470610efd823708"}}
      {"id":5,"name":"SH","cat":"dftracer","pid":3200613,"tid":3200613,"ph":"M","args":{"hhash":"f9ff883caaf21863","name":"DEFAULT-spawn","value":"8a4eff4d79020d73"}}
      {"id":6,"name":"start","cat":"dftracer","pid":3200613,"tid":3200613,"ts":1760934169561452,"dur":0,"ph":"X","args":{"hhash":"f9ff883caaf21863","cmd_hash":"b470610efd823708","p_idx":-1,"cwd":"457466652c169c22","level":1,"exec_hash":"8a4eff4d79020d73","version":"v1.0.15-56-gd5871bd","date":"Sun Oct 19 21:22:49 2025","ppid":3200524}}
      {"id":7,"name":"function.convolution.default","cat":"dynamo","pid":3200613,"tid":3200613,"ts":1760934170173819,"dur":22746,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"grad_enabled":0,"level":1}}
      {"id":8,"name":"function.relu.default","cat":"dynamo","pid":3200613,"tid":3200613,"ts":1760934170196754,"dur":289,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"grad_enabled":0,"level":1}}
      {"id":9,"name":"function.max_pool2d_with_indices.default","cat":"dynamo","pid":3200613,"tid":3200613,"ts":1760934170197063,"dur":10524,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"grad_enabled":0,"level":1}}
      {"id":10,"name":"function.getitem","cat":"dynamo","pid":3200613,"tid":3200613,"ts":1760934170207606,"dur":3,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"grad_enabled":0,"level":1}}
      {"id":11,"name":"function.getitem","cat":"dynamo","pid":3200613,"tid":3200613,"ts":1760934170207622,"dur":2,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"grad_enabled":0,"level":1}}
      {"id":12,"name":"function.view.default","cat":"dynamo","pid":3200613,"tid":3200613,"ts":1760934170207636,"dur":34,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"grad_enabled":0,"level":1}}
      {"id":13,"name":"function.t.default","cat":"dynamo","pid":3200613,"tid":3200613,"ts":1760934170207682,"dur":70,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"grad_enabled":0,"level":1}}
      {"id":14,"name":"function.addmm.default","cat":"dynamo","pid":3200613,"tid":3200613,"ts":1760934170207763,"dur":284,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"grad_enabled":0,"level":1}}
      {"id":15,"name":"end","cat":"dftracer","pid":3200613,"tid":3200613,"ts":1760934170208322,"dur":0,"ph":"X","args":{"hhash":"f9ff883caaf21863","p_idx":-1,"num_events":14,"level":1}}
  ]

Summary
-------

The Dynamo integration provides:

- **Deep PyTorch integration** via torch.compile
- **Operation-level tracing** for performance analysis
- **Minimal overhead** profiling
- **Easy to use** decorator-based API
- **Compatible** with existing PyTorch code

For complete API reference, see :doc:`api/dynamo`.
