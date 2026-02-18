utils.EditableHDF5Dataset.return_scalar
=======================================

Back to :doc:`Class page <../classes/utils.EditableHDF5Dataset>`

Summary
-------

Checks whether the return type is a scalar

Signature
---------

.. code-block:: python

   def return_scalar(self, key)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``key``
     - -
     - -
     - Key used to index into the dataset

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - True if the return type is a scalar

Docstring
---------

.. code-block:: text

   Checks whether the return type is a scalar
   :param key: Key used to index into the dataset
   :return: True if the return type is a scalar
