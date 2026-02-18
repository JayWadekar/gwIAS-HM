triggers_single_detector_HM.TriggerList.gen_triggers
====================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Defines a template bank with given spacing, performs matched filtering of data against this bank, and saves triggers above threshold

Signature
---------

.. code-block:: python

   def gen_triggers(self, delta_calpha = None, template_safety = None, remove_nonphysical = None, force_zero = None, threshold_chi2 = None, base_threshold_chi2 = None, trig_fname = None, config_fname = None, marginalized_score_HM = True, ncores = 1, n_wf_to_digest = None, subset = False, njobchunks = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``delta_calpha``
     - -
     - None
     - Spacing of template bank. Defaults to self.deltac_alpha if None
   * - ``template_safety``
     - -
     - None
     - Factor to inflate calpha range by. Defaults to self.template_safety if None
   * - ``remove_nonphysical``
     - -
     - None
     - Flag indicating whether to keep only the gridpoints that are close to a physical waveform. Defaults to self.remove_nonphysical if None
   * - ``force_zero``
     - -
     - None
     - Flag indicating whether to center the template grid to get as much overlap as possible with the central region. Defaults to self.force_zero if None
   * - ``threshold_chi2``
     - -
     - None
     - Threshold to save single-detector triggers above. Defaults to self.threshold_chi2 if None
   * - ``base_threshold_chi2``
     - -
     - None
     - Threshold to keep single-detector triggers for sinc interpolation. Defaults to self.base_threshold_chi2 if None
   * - ``trig_fname``
     - -
     - None
     - Absolute path to trigger file to save new triggers to, if desired 1. If trig_fname doesn't exist, we create it and save old processedclist + new processedclist to it 2. If trig_fname exists, we load it, and append new processedclist to it
   * - ``config_fname``
     - -
     - None
     - Name of configuration file to save trigger metadata
   * - ``marginalized_score_HM``
     - -
     - True
     - Use a semi-marginalized score instead of \\\|Z22\\\|\\\*\\\*2+\\\|Z33\\\|\\\*\\\*2+\\\|Z44\\\|\\\*\\\*2 as a lower threshold for storing triggers (basically, vetoes triggers with unphysical ratios of Amp33/Amp22 and Amp44/Amp22 close to the SNR threshold)
   * - ``ncores``
     - -
     - 1
     - Number of cores to parallelize over
   * - ``n_wf_to_digest``
     - -
     - None
     - Run only n_wf_to_digest waveforms per core
   * - ``subset``
     - -
     - False
     - Flag indicating whether to generate scores on a subset of data
   * - ``njobchunks``
     - -
     - 1
     - Number of chunks to divide job into, useful for restarting from an intermediate point

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Defines self.processedclist

Docstring
---------

.. code-block:: text

   Defines a template bank with given spacing, performs matched filtering
   of data against this bank, and saves triggers above threshold
   :param delta_calpha:
       Spacing of template bank. Defaults to self.deltac_alpha if None
   :param template_safety:
       Factor to inflate calpha range by. Defaults to self.template_safety
       if None
   :param remove_nonphysical:
       Flag indicating whether to keep only the gridpoints that are close
       to a physical waveform. Defaults to self.remove_nonphysical if None
   :param force_zero:
       Flag indicating whether to center the template grid to get as much
       overlap as possible with the central region. Defaults to
       self.force_zero if None
   :param threshold_chi2:
       Threshold to save single-detector triggers above. Defaults to
       self.threshold_chi2 if None
   :param base_threshold_chi2:
       Threshold to keep single-detector triggers for sinc interpolation.
       Defaults to self.base_threshold_chi2 if None
   :param trig_fname: Absolute path to trigger file to save new triggers
       to, if desired
       1. If trig_fname doesn't exist, we create it and save
          old processedclist + new processedclist to it
       2. If trig_fname exists, we load it, and append new processedclist
          to it
   :param config_fname: Name of configuration file to save trigger metadata
   :param marginalized_score_HM: Use a semi-marginalized score instead of 
       |Z22|**2+|Z33|**2+|Z44|**2 as a lower threshold for storing triggers
       (basically, vetoes triggers with unphysical ratios of Amp33/Amp22
        and Amp44/Amp22 close to the SNR threshold)
   :param ncores: Number of cores to parallelize over
   :param n_wf_to_digest: Run only n_wf_to_digest waveforms per core
   :param subset:
       Flag indicating whether to generate scores on a subset of data
   :param njobchunks:
       Number of chunks to divide job into, useful for restarting from an
       intermediate point
   :return: Defines self.processedclist
