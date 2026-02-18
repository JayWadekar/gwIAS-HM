triggers_single_detector_HM.TriggerList.prepare_for_triggers
============================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Set things up to generate triggers

Signature
---------

.. code-block:: python

   def prepare_for_triggers(self, average = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``average``
     - -
     - None
     - Method to compute the PSD drift correction (see gen_psd_drift_correction for options, defaults to class setup)

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

   Set things up to generate triggers
   :param average:
       Method to compute the PSD drift correction
       (see gen_psd_drift_correction for options, defaults to class setup)
   :return:
