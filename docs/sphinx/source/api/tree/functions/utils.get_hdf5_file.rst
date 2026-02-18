utils.get_hdf5_file
===================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Returns a hdf5 file object with the given mode, given a source. Warning, if the file is already open in read-only mode, we can't return a writeable version.

Signature
---------

.. code-block:: python

   def get_hdf5_file(source, mode = 'a', raise_error = False, **creation_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``source``
     - -
     - -
     - The path to the hdf5 object, or the File object itself
   * - ``mode``
     - -
     - 'a'
     - Mode to open the file in (recommended "r" for read only, and "a" for read/write)
   * - ``raise_error``
     - -
     - False
     - If True, raise an error if the file is not found
   * - ``\*\*creation_kwargs``
     - -
     - -
     - Keyword arguments to pass to h5py.File if creating it for the first time

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. File object that satisfies the mode conditions. Returns None if source is None, and reuses existing object if it satisfies the conditions 2. Boolean indicating if the file was opened in the function

Docstring
---------

.. code-block:: text

   Returns a hdf5 file object with the given mode, given a source. Warning, if
   the file is already open in read-only mode, we can't return a writeable
   version.
   :param source: The path to the hdf5 object, or the File object itself
   :param mode:
       Mode to open the file in (recommended "r" for read only, and "a" for
       read/write)
   :param raise_error: If True, raise an error if the file is not found
   :param creation_kwargs:
       Keyword arguments to pass to h5py.File if creating it for the first time
   :return:
       1. File object that satisfies the mode conditions. Returns None if
          source is None, and reuses existing object if it satisfies
          the conditions
       2. Boolean indicating if the file was opened in the function
