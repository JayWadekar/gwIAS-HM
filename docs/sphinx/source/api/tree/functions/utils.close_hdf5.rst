utils.close_hdf5
================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def close_hdf5(fnames = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fnames``
     - -
     - None
     - If known, close only these filenames (single or list)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Closes open hdf5 files (all, or selected)

Docstring
---------

.. code-block:: text

   :param fnames: If known, close only these filenames (single or list)
   :return: Closes open hdf5 files (all, or selected)
