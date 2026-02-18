utils.save_dict_to_hdf5_attrs
=============================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Saves a dictionary to the attributes of a hdf5 file

Signature
---------

.. code-block:: python

   def save_dict_to_hdf5_attrs(fobj, dic, keys_to_skip = None, overwrite = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fobj``
     - -
     - -
     - hdf5 file or group object
   * - ``dic``
     - -
     - -
     - Dictionary to save
   * - ``keys_to_skip``
     - -
     - None
     - List of keys to skip
   * - ``overwrite``
     - -
     - False
     - If True, overwrite existing attributes

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - None

Docstring
---------

.. code-block:: text

   Saves a dictionary to the attributes of a hdf5 file
   :param fobj: hdf5 file or group object
   :param dic: Dictionary to save
   :param keys_to_skip: List of keys to skip
   :param overwrite: If True, overwrite existing attributes
   :return: None
