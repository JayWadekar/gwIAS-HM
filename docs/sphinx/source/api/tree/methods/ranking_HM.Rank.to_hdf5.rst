ranking_HM.Rank.to_hdf5
=======================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

Saves the class to a hdf5 file. Ensure that you ran check_and_fix_bg_fg_lists if you added extra injections (scoring does it automatically) Note that saving to a new path doesn't replace what is in the instance unless path is None, if you want to use the new hdf5 file created, load it using from_hdf5

Signature
---------

.. code-block:: python

   def to_hdf5(self, path = None, overwrite = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``path``
     - -
     - None
     - Path to the hdf5 file to save to, can also be a HDF5 File object. If None, defaults to self.fobj
   * - ``overwrite``
     - -
     - False
     - Flag to overwrite data in the file if it exists

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

   Saves the class to a hdf5 file. Ensure that you ran
   check_and_fix_bg_fg_lists if you added extra injections (scoring does
   it automatically)
   Note that saving to a new path doesn't replace what is in the instance
   unless path is None, if you want to use the new hdf5 file created, load
   it using from_hdf5
   :param path:
       Path to the hdf5 file to save to, can also be a HDF5 File object.
       If None, defaults to self.fobj
   :param overwrite: Flag to overwrite data in the file if it exists
   :return:
