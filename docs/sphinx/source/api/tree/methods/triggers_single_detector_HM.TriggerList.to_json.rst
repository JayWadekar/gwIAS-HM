triggers_single_detector_HM.TriggerList.to_json
===============================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Saves information about trigger list to json file

Signature
---------

.. code-block:: python

   def to_json(self, config_fname, preprocessing_fname = None, overwrite = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``config_fname``
     - -
     - -
     - Absolute path of json file
   * - ``preprocessing_fname``
     - -
     - None
     - Override preprocessing file, pass without extension
   * - ``overwrite``
     - -
     - False
     - Flag to overwrite existing preprocessing file

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

   Saves information about trigger list to json file
   :param config_fname: Absolute path of json file
   :param preprocessing_fname:
       Override preprocessing file, pass without extension
   :param overwrite: Flag to overwrite existing preprocessing file
   :return:
