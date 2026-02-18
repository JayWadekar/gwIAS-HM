triggers_single_detector_HM.TriggerList.finer_psd_drift
=======================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def finer_psd_drift(self, trigger, wf_whitened_fd = None, average = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - -
     - Row of processedclist
   * - ``wf_whitened_fd``
     - -
     - None
     - Frequency domain waveform to use for PSD drift. If None, we use the waveform corresponding to the calphas in trigger
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
     - Correction factor to multiply SNR^2, equals 1 if finer PSD drift correction is consistent with coarser one

Docstring
---------

.. code-block:: text

   :param trigger: Row of processedclist
   :param wf_whitened_fd:
       Frequency domain waveform to use for PSD drift. If None,
       we use the waveform corresponding to the calphas in trigger
   :param average:
       Method to compute the PSD drift correction
       (see gen_psd_drift_correction for options, defaults to class setup)
   :return: Correction factor to multiply SNR^2, equals 1 if finer
            PSD drift correction is consistent with coarser one
