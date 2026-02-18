triggers_single_detector_HM.ensure_abspath
==========================================

Back to :doc:`Module page <../modules/triggers_single_detector_HM>`

Summary
-------

Ensures that the path to fname is absolute, assumes it is saved in the same directory as config_fname!

Signature
---------

.. code-block:: python

   def ensure_abspath(config_fname, fname)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``config_fname``
     - -
     - -
     - Absolute path to config.json file
   * - ``fname``
     - -
     - -
     - Absolute or relative path to file

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Absolute path to fname

Docstring
---------

.. code-block:: text

   Ensures that the path to fname is absolute, assumes it is saved
   in the same directory as config_fname!
   :param config_fname: Absolute path to config.json file
   :param fname: Absolute or relative path to file
   :return: Absolute path to fname
