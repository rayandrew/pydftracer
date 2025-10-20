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

Utility Functions
-----------------

.. autofunction:: dftracer.python.dynamo.create_detailed_op_name

Usage Examples
--------------

See :doc:`../dynamo_guide` for detailed usage examples and best practices.
