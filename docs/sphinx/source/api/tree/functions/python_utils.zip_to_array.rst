python_utils.zip_to_array
=========================

Back to :doc:`Module page <../modules/python_utils>`

Summary
-------

O(10^3)x faster np.array([row for row in np.broadcast(\\\*args)]) usable when args are all at most 1d.

Signature
---------

.. code-block:: python

   def zip_to_array(*args)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``\*args``
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

   O(10^3)x faster np.array([row for row in np.broadcast(*args)])
   usable when args are all at most 1d.
