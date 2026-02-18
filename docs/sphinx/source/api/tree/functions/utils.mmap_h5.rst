utils.mmap_h5
=============

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Memory mapping is much faster for single row access than non-chunked hdf5, as long as we have a 64 bit architecture It is slower for random access, as the kernel version on the IAS systems is older and doesn't activate the madvise system call WARNING: If the hdf5 file is already open (e.g., in a writeable manner), the values in the mmap object might not be updated until the changes to the hdf5 file are flushed (despite the description of the dedault UNIX driver, it has a small buffer, set by rdcc_nbytes)!

Signature
---------

.. code-block:: python

   def mmap_h5(path, h5path, mode = 'r', dtype = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``path``
     - -
     - -
     - Path to hdf5 file, or the file object itself
   * - ``h5path``
     - -
     - -
     - Name of dataset within the file, or iterable with path to it
   * - ``mode``
     - -
     - 'r'
     - {‘r+’, ‘r’, ‘w+’, ‘c’}, see documentation of mmap
   * - ``dtype``
     - -
     - None
     - Interpret the data using this datatype (defaults to the saved type, different from type conversion! use only for complex/real)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - mmap object to query as needed

Docstring
---------

.. code-block:: text

   Memory mapping is much faster for single row access than non-chunked hdf5,
   as long as we have a 64 bit architecture
   It is slower for random access, as the kernel version on the IAS systems
   is older and doesn't activate the madvise system call
   WARNING: If the hdf5 file is already open (e.g., in a writeable manner), the
   values in the mmap object might not be updated until the changes to the hdf5
   file are flushed (despite the description of the dedault UNIX driver, it has
   a small buffer, set by rdcc_nbytes)!
   :param path: Path to hdf5 file, or the file object itself
   :param h5path: Name of dataset within the file, or iterable with path to it
   :param mode: {‘r+’, ‘r’, ‘w+’, ‘c’}, see documentation of mmap
   :param dtype:
       Interpret the data using this datatype (defaults to the saved type,
       different from type conversion! use only for complex/real)
   :return: mmap object to query as needed
