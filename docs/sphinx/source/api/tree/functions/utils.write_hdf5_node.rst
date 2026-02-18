utils.write_hdf5_node
=====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Convenience function to create/overwrite a leaf in a hdf5 file

Signature
---------

.. code-block:: python

   def write_hdf5_node(fobj, h5path, data, dtype = None, overwrite = True, outformat = h5py.Dataset, outdtype = None, outmode = 'r+')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fobj``
     - -
     - -
     - hdf5 file object
   * - ``h5path``
     - -
     - -
     - String, or an iterable of strings to reach the leaf. The last entry is the leaf's name
   * - ``data``
     - -
     - -
     - Data to write in the leaf
   * - ``dtype``
     - -
     - None
     - If known, type of data. The default of None is to infer it
   * - ``overwrite``
     - -
     - True
     - Flag indicating whether to overwrite the leaf if it exists 0: Returns the existing key without doing anything 1: Deletes the existing dataset and creates a new one 2. Uses the existing and allocated space if it is possible If boolean, False = 0, True = 2
   * - ``outformat``
     - -
     - h5py.Dataset
     - Format of object to create (default is dataset, can also be mmap). It can be a string or a type.
   * - ``outdtype``
     - -
     - None
     - If known, how to read the data (only used for mmap)
   * - ``outmode``
     - -
     - 'r+'
     - Mode to open the file in (only used for mmap)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - The leaf object in the desired format

Docstring
---------

.. code-block:: text

   Convenience function to create/overwrite a leaf in a hdf5 file
   :param fobj: hdf5 file object
   :param h5path:
       String, or an iterable of strings to reach the leaf. The last entry is
       the leaf's name
   :param data: Data to write in the leaf
   :param dtype: If known, type of data. The default of None is to infer it
   :param overwrite:
       Flag indicating whether to overwrite the leaf if it exists
       0: Returns the existing key without doing anything
       1: Deletes the existing dataset and creates a new one
       2. Uses the existing and allocated space if it is possible
       If boolean, False = 0, True = 2
   :param outformat:
       Format of object to create (default is dataset, can also be mmap).
       It can be a string or a type.
   :param outdtype: If known, how to read the data (only used for mmap)
   :param outmode: Mode to open the file in (only used for mmap)
   :return: The leaf object in the desired format
