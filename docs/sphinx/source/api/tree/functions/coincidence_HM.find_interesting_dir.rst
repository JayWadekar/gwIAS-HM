coincidence_HM.find_interesting_dir
===================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Goes through trigger files for H1 and L1 in a directory and performs coincidence analysis

Signature
---------

.. code-block:: python

   def find_interesting_dir(dir_name, enumerated_epochs = None, n_epochs = None, time_shift_tol = 0.01, score_reduction_max = 5, threshold_chi2 = 60.0, max_time_slide_shift = 100, minimal_time_slide_jump = 0.1, min_veto_chi2 = 30, max_zero_lag_delay = 0.015, out_fname = None, n_cores = 1, run = 'O3a', opt_format = 'new', outfile_format = 'new', old_cand_dir_name = None, bad_times = None, veto_triggers = True, output_timeseries = True, output_coherent_score = True, score_reduction_timeseries = 10, detectors = ('H1', 'L1'), weaker_detectors = (), recompute_psd_drift = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dir_name``
     - -
     - -
     - Path to a directory with json files for H1 and L1
   * - ``enumerated_epochs``
     - -
     - None
     - If desired, list of integer epochs to analyze
   * - ``n_epochs``
     - -
     - None
     - If desired, restrict to this number of H1 epochs
   * - ``time_shift_tol``
     - -
     - 0.01
     - Width (s) of buckets to collect triggers into, the \\\`friends" of a trigger with the same calpha live within the same bucket
   * - ``score_reduction_max``
     - -
     - 5
     - Absolute reduction in SNR^2 from the peak value in each bucket to retain (we also have a hardcoded relative reduction)
   * - ``threshold_chi2``
     - -
     - 60.0
     - Threshold in sum(SNR^2) above which we consider triggers for the background list (or signal)
   * - ``max_time_slide_shift``
     - -
     - 100
     - Max delay allowed for background triggers
   * - ``minimal_time_slide_jump``
     - -
     - 0.1
     - Jumps in timeslides
   * - ``min_veto_chi2``
     - -
     - 30
     - Apply vetos to candidates above this SNR^2 in a single detector
   * - ``max_zero_lag_delay``
     - -
     - 0.015
     - Maximum delay between detectors within the same timeslide
   * - ``out_fname``
     - -
     - None
     - Path to npy file to save the candidates to
   * - ``n_cores``
     - -
     - 1
     - Number of cores to use for splitting the veto computations
   * - ``run``
     - -
     - 'O3a'
     - String identifying the run
   * - ``opt_format``
     - -
     - 'new'
     - How we choose the finer calpha grid, changed between O1 and O2 analyses Exposed here to replicate old runs if needed
   * - ``outfile_format``
     - -
     - 'new'
     - FLag whether to save the old style (separate npy files for different arrays), or in the new style with a consolidated file per job
   * - ``old_cand_dir_name``
     - -
     - None
     - Directory with old vetoed candidate files, if we want to save on veto computations when redoing
   * - ``bad_times``
     - -
     - None
     - List of lists of times to avoid in H1 and L1, if known
   * - ``veto_triggers``
     - -
     - True
     - Flag to turn the veto on/off
   * - ``output_timeseries``
     - -
     - True
     - Flag to output timeseries for the candidates
   * - ``output_coherent_score``
     - -
     - True
     - Flag to compute the coherent score integral for the candidates
   * - ``score_reduction_timeseries``
     - -
     - 10
     - Restrict triggers in timeseries to the ones with single_detector_SNR^2 > (base trigger SNR^2) - this parameter
   * - ``detectors``
     - -
     - ('H1', 'L1')
     - Tuple with names of the two detectors we will be running coincidence with ('H1', 'L1', 'V1' supported)
   * - ``weaker_detectors``
     - -
     - ()
     - If needed, tuple with names of weaker detectors that we will compute timeseries for as well ('H1', 'L1', 'V1' supported) Note: Only works with outfile_format == "new"
   * - ``recompute_psd_drift``
     - -
     - False
     - Flag to recompute PSD drift correction. We needed it in O2 since the trigger files didn't use safemean. Redundant in O3a and forwards

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

   Goes through trigger files for H1 and L1 in a directory and performs
   coincidence analysis
   :param dir_name: Path to a directory with json files for H1 and L1
   :param enumerated_epochs: If desired, list of integer epochs to analyze
   :param n_epochs: If desired, restrict to this number of H1 epochs
   :param time_shift_tol:
       Width (s) of buckets to collect triggers into, the `friends" of a
       trigger with the same calpha live within the same bucket
   :param score_reduction_max:
       Absolute reduction in SNR^2 from the peak value in each bucket to
       retain (we also have a hardcoded relative reduction)
   :param threshold_chi2:
       Threshold in sum(SNR^2) above which we consider triggers for the
       background list (or signal)
   :param max_time_slide_shift: Max delay allowed for background triggers
   :param minimal_time_slide_jump: Jumps in timeslides
   :param min_veto_chi2:
       Apply vetos to candidates above this SNR^2 in a single detector
   :param max_zero_lag_delay:
       Maximum delay between detectors within the same timeslide
   :param out_fname: Path to npy file to save the candidates to
   :param n_cores: Number of cores to use for splitting the veto computations
   :param run: String identifying the run
   :param opt_format:
       How we choose the finer calpha grid, changed between O1 and O2 analyses
       Exposed here to replicate old runs if needed
   :param outfile_format:
       FLag whether to save the old style (separate npy files for different
       arrays), or in the new style with a consolidated file per job
   :param old_cand_dir_name:
       Directory with old vetoed candidate files, if we want to save on veto
       computations when redoing
   :param bad_times: List of lists of times to avoid in H1 and L1, if known
   :param veto_triggers: Flag to turn the veto on/off
   :param output_timeseries: Flag to output timeseries for the candidates
   :param output_coherent_score:
       Flag to compute the coherent score integral for the candidates
   :param score_reduction_timeseries:
       Restrict triggers in timeseries to the ones with
       single_detector_SNR^2 > (base trigger SNR^2) - this parameter
   :param detectors:
       Tuple with names of the two detectors we will be running coincidence
       with ('H1', 'L1', 'V1' supported)
   :param weaker_detectors:
       If needed, tuple with names of weaker detectors that we will compute
       timeseries for as well ('H1', 'L1', 'V1' supported)
       Note: Only works with outfile_format == "new"
   :param recompute_psd_drift:
       Flag to recompute PSD drift correction. We needed it in O2 since the
       trigger files didn't use safemean. Redundant in O3a and forwards
   :return:
