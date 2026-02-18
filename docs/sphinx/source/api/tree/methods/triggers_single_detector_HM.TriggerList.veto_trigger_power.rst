triggers_single_detector_HM.TriggerList.veto_trigger_power
==========================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Checks trigger by subtracting best fit waveform and looking for excess power commensurate with waveform

Signature
---------

.. code-block:: python

   def veto_trigger_power(self, trigger, template_max_mismatch = params.DEF_TEMPLATE_MAX_MISMATCH, nfire = params.NFIRE, subset_defined = False, bestfit_wf = None, test_injection = False, lazy = True, physical_mode_ratio = False, verbose = 0)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - -
     - Row in TriggerList.processedclist corresp. to trigger
   * - ``template_max_mismatch``
     - -
     - params.DEF_TEMPLATE_MAX_MISMATCH
     - Reduce the preserve max snr by this factor
   * - ``nfire``
     - -
     - params.NFIRE
     - Number of times glitch detectors fire per perfect file
   * - ``subset_defined``
     - -
     - False
     - Flag indicating we have already defined appropriate subset of data
   * - ``bestfit_wf``
     - -
     - None
     - If known, pass the bestfit waveform (we can use any other waveform, as long as we take care of the indices)
   * - ``test_injection``
     - -
     - False
     - Use the injected waveform as the bestfit waveform
   * - ``lazy``
     - -
     - True
     - Flag indicating whether to stop after a test fails
   * - ``physical_mode_ratio``
     - -
     - False
     - require A_33/A_22 and A_44/A_22 of the best-fit wf to be in physically allowed space
   * - ``verbose``
     - -
     - 0
     - Integer flag indicating level of details to print 0. Do not print anything 1. Print which test failed 2. Print which test failed, and details of the test

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 0. True/False according to whether the trigger passed/failed 1. Correction factor to multiply SNR^2, different from 1 if finer PSD drift correction is significantly different 2. Boolean array of size len(self.outlier_reasons) with zeros marking glitch tests that fired The indices correspond to: 0: len(self.outlier_reasons)-1: (index into outlier reasons)-1 for excess-power-like tests (outlier reasons[0] doesn't happen here) len(self.outlier_reasons)-1: Finer PSD drift killed it

Docstring
---------

.. code-block:: text

   Checks trigger by subtracting best fit waveform and looking for excess
   power commensurate with waveform
   :param trigger: Row in TriggerList.processedclist corresp. to trigger
   :param template_max_mismatch: Reduce the preserve max snr by this factor
   :param nfire: Number of times glitch detectors fire per perfect file
   :param subset_defined:
       Flag indicating we have already defined appropriate subset of data
   :param bestfit_wf:
       If known, pass the bestfit waveform (we can use any other waveform,
       as long as we take care of the indices)
   :param test_injection: Use the injected waveform as the bestfit waveform
   :param lazy: Flag indicating whether to stop after a test fails
   :param physical_mode_ratio: require A_33/A_22 and A_44/A_22 of the best-fit wf
                               to be in physically allowed space
   :param verbose:
       Integer flag indicating level of details to print
       0. Do not print anything
       1. Print which test failed
       2. Print which test failed, and details of the test
   :return: 0. True/False according to whether the trigger passed/failed
            1. Correction factor to multiply SNR^2, different from 1 if
               finer PSD drift correction is significantly different
            2. Boolean array of size len(self.outlier_reasons) with zeros
               marking glitch tests that fired
               The indices correspond to:
               0: len(self.outlier_reasons)-1:
                   (index into outlier reasons)-1 for excess-power-like
                   tests (outlier reasons[0] doesn't happen here)
               len(self.outlier_reasons)-1: Finer PSD drift killed it
