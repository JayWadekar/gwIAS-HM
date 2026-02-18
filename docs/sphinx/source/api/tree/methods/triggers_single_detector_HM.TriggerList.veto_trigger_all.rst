triggers_single_detector_HM.TriggerList.veto_trigger_all
========================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Convenience function to apply all vetos

Signature
---------

.. code-block:: python

   def veto_trigger_all(self, trigger, dcalphas = None, short_wf_limit = params.SHORT_WF_LIMIT, subset_details = None, verbose = False, lazy = True, do_costly_vetos = True, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - -
     - Arrow of length len(processedclist[0]) with trigger
   * - ``dcalphas``
     - -
     - None
     - Array with spacings in calphas to allow mismatch in
   * - ``short_wf_limit``
     - -
     - params.SHORT_WF_LIMIT
     - Waveform duration below which we avoid holes
   * - ``subset_details``
     - -
     - None
     - If available, output of prepare_subset_for_vetoes
   * - ``verbose``
     - -
     - False
     - Flag indicating whether to print details
   * - ``lazy``
     - -
     - True
     - Flag indicating whether to stop after a test fails
   * - ``do_costly_vetos``
     - -
     - True
     - Flag indicating whether we are doing veto_trigger_phase and veto_trigger_power
   * - ``\*\*kwargs``
     - -
     - -
     - Arguments to pass to vetos

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Flag that is true/false if the trigger passed/failed 2. Correction factor to multiply snr^2 due to finer PSD drift 3. Boolean array of size len(self.outlier_reasons) + 3 + len(split_chunks) with zeros marking glitch tests that fired The indices correspond to: 0: len(self.outlier_reasons): index into outlier reasons for excess-power-like tests len(self.outlier_reasons): Finer PSD drift killed it len(self.outlier_reasons) + 1: No chunks present len(self.outlier_reasons) + 2: Overall chi-2 test len(self.outlier_reasons) + 3: len(self.outlier_reasons) + 3 + len(split_chunks): Split tests

Docstring
---------

.. code-block:: text

   Convenience function to apply all vetos
   :param trigger: Arrow of length len(processedclist[0]) with trigger
   :param dcalphas: Array with spacings in calphas to allow mismatch in
   :param short_wf_limit: Waveform duration below which we avoid holes
   :param subset_details: If available, output of prepare_subset_for_vetoes
   :param verbose: Flag indicating whether to print details
   :param lazy: Flag indicating whether to stop after a test fails
   :param do_costly_vetos: Flag indicating whether we are doing 
                           veto_trigger_phase and veto_trigger_power
   :param kwargs: Arguments to pass to vetos
   :return: 1. Flag that is true/false if the trigger passed/failed
            2. Correction factor to multiply snr^2 due to finer PSD drift
            3. Boolean array of size
               len(self.outlier_reasons) + 3 + len(split_chunks)
               with zeros marking glitch tests that fired
               The indices correspond to:
               0: len(self.outlier_reasons): index into outlier reasons
                  for excess-power-like tests
               len(self.outlier_reasons): Finer PSD drift killed it
               len(self.outlier_reasons) + 1: No chunks present
               len(self.outlier_reasons) + 2: Overall chi-2 test
               len(self.outlier_reasons) + 3:
                   len(self.outlier_reasons) + 3 + len(split_chunks):
                       Split tests
