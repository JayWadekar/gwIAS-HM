python_utils.zeropad_end
========================

Back to :doc:`Module page <../modules/python_utils>`

Summary
-------

pad arr axis=-1 with zeros at (right/back) end to length pad_to_N if arr.shape[-1] < pad_to_N, raise ValueError

Signature
---------

.. code-block:: python

   def zeropad_end(arr, pad_to_N)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``arr``
     - -
     - -
     - -
   * - ``pad_to_N``
     - -
     - -
     - -

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - -

Docstring
---------

.. code-block:: text

   pad arr axis=-1 with zeros at (right/back) end to length pad_to_N
     if arr.shape[-1] < pad_to_N, raise ValueError
