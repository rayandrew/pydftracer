Dynamo API
==========

This section documents the PyTorch Dynamo integration for tracing compiled models.

Dynamo Class
------------

.. autoclass:: dftracer.python.Dynamo
   :members:
   :undoc-members:
   :show-inheritance:

Module Instance
---------------

The module provides a pre-configured ``dynamo`` instance:

.. code-block:: python

   from dftracer.python import dynamo

   # Use the dynamo decorator
   @dynamo.compile
   def forward(self, x):
       return x * 2

Constants
---------

.. autodata:: dftracer.python.dynamo.CAT_DYNAMO
   :annotation: = "dynamo"

Internal Classes
----------------

CallStackRecord
~~~~~~~~~~~~~~~

.. autoclass:: dftracer.python.dynamo.CallStackRecord
   :members:
   :undoc-members:

Backend Functions
-----------------

create_backend
~~~~~~~~~~~~~~

.. autofunction:: dftracer.python.dynamo.create_backend

The ``create_backend`` function creates a custom PyTorch compile backend with DFTracer instrumentation.
This allows you to use DFTracer directly with ``torch.compile()``.

**Example:**

.. code-block:: python

   from dftracer.python.dynamo import create_backend
   import torch

   # Create a backend with custom parameters
   backend = create_backend(
       name="my_model",
       epoch=0,
       step=0,
       enable=True,
       autograd=True
   )

   # Use with torch.compile
   model = MyModel()
   compiled_model = torch.compile(model, backend=backend)

Utility Functions
-----------------

create_detailed_op_name
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dftracer.python.dynamo.create_detailed_op_name

Usage Examples
--------------

See :doc:`../dynamo_guide` for detailed usage examples and best practices.
