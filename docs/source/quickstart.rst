Quick Start
===========

This guide will help you get started with pydftracer quickly.

Basic Usage
-----------

Enabling DFTracer
~~~~~~~~~~~~~~~~~

DFTracer must be enabled via environment variable before use:

.. code-block:: bash

   export DFTRACER_ENABLE=1
   python your_script.py

Initialize DFTracer Logger
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, initialize the DFTracer logger:

.. code-block:: python

   from dftracer.python import dftracer

   # Initialize with log file path, data directory, and process ID
   df_logger = dftracer.initialize_log(
       log_file="trace.pfw",
       data_dir="/tmp/data",
       process_id=-1  # -1 for auto process ID
   )

   # Your code here - I/O operations will be automatically traced

   # Always finalize when done
   df_logger.finalize()

Function Tracing with dft_fn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``dft_fn`` to create tracers for specific functions:

.. code-block:: python

   from dftracer.python import dftracer, dft_fn
   import numpy as np

   # Initialize logger
   df_logger = dftracer.initialize_log("trace.pfw", "/tmp/data", -1)

   # Create a function tracer
   io_tracer = dft_fn("data_io")

   @io_tracer.log
   def write_data(filename, data):
       np.save(filename, data)

   @io_tracer.log
   def read_data(filename):
       return np.load(filename)

   # Use the traced functions
   data = np.ones((100, 100))
   write_data('test.npy', data)
   result = read_data('test.npy')

   df_logger.finalize()

Using Iterators
~~~~~~~~~~~~~~~

Track iterations with ``dft_fn.iter()``:

.. code-block:: python

   from dftracer.python import dftracer, dft_fn

   df_logger = dftracer.initialize_log("trace.pfw", "/tmp/data", -1)
   my_tracer = dft_fn("training")

   @my_tracer.log
   def process_batch(batch_id):
       # Process each batch
       for i in my_tracer.iter(range(10)):
           # Process item i
           pass

   process_batch(0)
   df_logger.finalize()

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Configure pydftracer using environment variables:

.. code-block:: bash

   # Enable DFTracer
   export DFTRACER_ENABLE=1

   # Set initialization mode (PRELOAD or other)
   export DFTRACER_INIT=PRELOAD

   # Set log level (DEBUG, INFO, WARN, ERROR)
   export DFTRACER_LOG_LEVEL=INFO

You can also check these in your code:

.. code-block:: python

   from dftracer.python import (
       DFTRACER_ENABLE,
       DFTRACER_INIT_PRELOAD,
       DFTRACER_LOG_LEVEL
   )

   if DFTRACER_ENABLE:
       print("DFTracer is enabled")

AI/ML Tracing
-------------

PyTorch Dynamo Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For PyTorch applications, use the Dynamo backend:

.. code-block:: python

   import torch
   from dftracer.python import dftracer_dynamo_backend

   # Use DFTracer as a PyTorch compile backend
   model = MyModel()
   compiled_model = torch.compile(model, backend=dftracer_dynamo_backend)

   # Run your model
   output = compiled_model(input_tensor)

Dynamo Class
~~~~~~~~~~~~

For more control over Dynamo tracing:

.. code-block:: python

   from dftracer.python import Dynamo

   # Create a Dynamo tracer instance
   dynamo_tracer = Dynamo(
       name="my_model",
       epoch=1,
       step=100,
       enable=True
   )

   # Use in your training loop
   # The tracer will record PyTorch operations

AI Tracing Features
~~~~~~~~~~~~~~~~~~~

Use the AI tracing utilities:

.. code-block:: python

   from dftracer.python import DFTracerAI

   # Create an AI-specific tracer
   ai_tracer = DFTracerAI(
       cat="training",
       name="resnet50",
       epoch=5,
       step=1000,
       enable=True
   )

Advanced Usage
--------------

Custom Tags
~~~~~~~~~~~

Add custom metadata to your traces:

.. code-block:: python

   from dftracer.python import dftracer, TagValue, TagDType, TagType

   dft = dftracer()

   # Create custom tags
   tag = TagValue(
       value="my_value",
       dtype=TagDType.STRING,
       tag_type=TagType.KEY
   )

   # Use in your traced functions

Metadata Events
~~~~~~~~~~~~~~~

Log metadata events:

.. code-block:: python

   from dftracer.python import dftracer

   dft = dftracer()

   # Log metadata
   dft.log_metadata_event("key", "value")

Next Steps
----------

- Explore the :doc:`api/index` for detailed API documentation
- Check the `DFTracer main documentation <https://dftracer.readthedocs.io/>`_ for more advanced features
- Look at example scripts in the repository
