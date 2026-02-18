utils.load_dict_from_hdf5_attrs
===============================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Loads a dictionary from the attributes of a hdf5 file

Signature
---------

.. code-block:: python

   def load_dict_from_hdf5_attrs(fobj, keys = None, outdict = None)

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
   * - ``keys``
     - -
     - None
     - List of keys to load. If None, load all keys
   * - ``outdict``
     - -
     - None
     - Dictionary to load into, if None, create a new one

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Dictionary with the keys and values if outdict is None, else modifies what was passed

Docstring
---------

.. code-block:: text

   Loads a dictionary from the attributes of a hdf5 file
   :param fobj: hdf5 file or group object
   :param keys: List of keys to load. If None, load all keys
   :param outdict: Dictionary to load into, if None, create a new one
   :return:
       Dictionary with the keys and values if outdict is None, else modifies
       what was passed
