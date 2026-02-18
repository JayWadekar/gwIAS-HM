utils.read_hdf5_node
====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Convenience function to read/create a group from a hdf5 file

Signature
---------

.. code-block:: python

   def read_hdf5_node(fobj, h5path, create = True, raise_error = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fobj``
     - -
     - -
     - hdf5 file or group
   * - ``h5path``
     - -
     - -
     - String, or an iterable of strings to reach the leaf
   * - ``create``
     - -
     - True
     - If True, we create what doesn't exist
   * - ``raise_error``
     - -
     - False
     - If True, raise an error if the path doesn't exist

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - group object, or None if it doesn't exist and create and raise_error are False

Docstring
---------

.. code-block:: text

   Convenience function to read/create a group from a hdf5 file
   :param fobj: hdf5 file or group
   :param h5path: String, or an iterable of strings to reach the leaf
   :param create: If True, we create what doesn't exist
   :param raise_error: If True, raise an error if the path doesn't exist
   :return:
       group object, or None if it doesn't exist and create and raise_error
       are False
