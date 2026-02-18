utils.get_strain_fnames
=======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Returns the strain file names given a time

Signature
---------

.. code-block:: python

   def get_strain_fnames(t, run = None, check_exists = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``t``
     - -
     - -
     - GPS time
   * - ``run``
     - -
     - None
     - String identifying the run, if None, infers from the GPS time
   * - ``check_exists``
     - -
     - True
     - If True, returns None for missing files

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - List of length n_detectors with strain file names

Docstring
---------

.. code-block:: text

   Returns the strain file names given a time
   :param t: GPS time
   :param run: String identifying the run, if None, infers from the GPS time
   :param check_exists: If True, returns None for missing files
   :return: List of length n_detectors with strain file names
