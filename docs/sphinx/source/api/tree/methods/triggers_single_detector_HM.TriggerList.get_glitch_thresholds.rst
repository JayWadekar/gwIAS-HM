triggers_single_detector_HM.TriggerList.get_glitch_thresholds
=============================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Function giving test-statistic values for various glitch tests attained in the presence of noiseless waveforms

Signature
---------

.. code-block:: python

   def get_glitch_thresholds(wt_wfs_cos, dt, preserve_max_snr, sine_gaussian_intervals, bandlim_transient_intervals, excess_power_intervals, include_HM_wfs = False, bank = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wt_wfs_cos``
     - -
     - -
     - Whitened waveform(s) with SNR = 1 Array of size: nwfs x fftsize Combine the modes first if you want to pass a waveform with HM
   * - ``dt``
     - -
     - -
     - Sampling interval (s)
   * - ``preserve_max_snr``
     - -
     - -
     - We assure that waveforms with this SNR are preserved
   * - ``sine_gaussian_intervals``
     - -
     - -
     - -
   * - ``bandlim_transient_intervals``
     - -
     - -
     - -
   * - ``excess_power_intervals``
     - -
     - -
     - -
   * - ``include_HM_wfs``
     - -
     - False
     - Boolean flag whether to add to wt_wfs_cos specific waveforms simulated using parameters from template bank metadata
   * - ``bank``
     - -
     - None
     - Instance of TemplateBank

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Threshold, and index for sigma clipping 2. Thresholds, and indices for sine gaussian transients 3. Thresholds, and indices for bandlimited power transients 4. Thresholds, and indices for excess power

Docstring
---------

.. code-block:: text

   Function giving test-statistic values for various glitch tests attained
   in the presence of noiseless waveforms
   :param wt_wfs_cos: Whitened waveform(s) with SNR = 1
                      Array of size: nwfs x fftsize
                      Combine the modes first if you want to pass a waveform with HM
   :param dt: Sampling interval (s)
   :param preserve_max_snr:
       We assure that waveforms with this SNR are preserved
   :param sine_gaussian_intervals:
   :param bandlim_transient_intervals:
   :param excess_power_intervals:
   :param include_HM_wfs: Boolean flag whether to add to wt_wfs_cos specific 
                          waveforms simulated using parameters from
                          template bank metadata
   :param bank: Instance of TemplateBank
   :return: 1. Threshold, and index for sigma clipping
            2. Thresholds, and indices for sine gaussian transients
            3. Thresholds, and indices for bandlimited power transients
            4. Thresholds, and indices for excess power
