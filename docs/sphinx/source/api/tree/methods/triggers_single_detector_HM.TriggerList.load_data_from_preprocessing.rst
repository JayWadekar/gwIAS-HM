triggers_single_detector_HM.TriggerList.load_data_from_preprocessing
====================================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Load preprocessed data from file

Signature
---------

.. code-block:: python

   def load_data_from_preprocessing(self, fname_preprocessing, do_ffts = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fname_preprocessing``
     - -
     - -
     - Filename to load preprocessed data from
   * - ``do_ffts``
     - -
     - True
     - Flag indicating whether to define whitening filter If boolean, True/False If integer, >0 we define the whitening filter

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

   Load preprocessed data from file
   :param fname_preprocessing: Filename to load preprocessed data from
   :param do_ffts:
       Flag indicating whether to define whitening filter
       If boolean, True/False
       If integer, >0 we define the whitening filter
   :return:
