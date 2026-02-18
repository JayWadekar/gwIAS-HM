coincidence_HM.find_interesting_dir_cluster
===========================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def find_interesting_dir_cluster(dir_path, output_dir = None, threshold_chi2 = 60.0, min_veto_chi2 = 30, max_time_slide_shift = 100, score_reduction_max = 5, time_shift_tol = 0.01, minimal_time_slide_jump = 0.1, max_zero_lag_delay = 0.015, opt_format = 'new', output_timeseries = True, output_coherent_score = True, score_reduction_timeseries = 10, detectors = ('H1', 'L1'), weaker_detectors = (), recompute_psd_drift = False, outfile_format = 'new', job_name = 'coincidence', n_cores = 1, n_hours_limit = 12, epoch_list = None, n_jobs = 256, mem_limit = None, debug = False, submit = False, overwrite = False, cluster = 'typhon', exclusive = False, run = 'O3a', cver = 6, rerun = False, old_cver = None, exclude_nodes = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dir_path``
     - -
     - -
     - Path with output files of the triggering
   * - ``output_dir``
     - -
     - None
     - Where to place the coincident_files, if known
   * - ``threshold_chi2``
     - -
     - 60.0
     - -
   * - ``min_veto_chi2``
     - -
     - 30
     - -
   * - ``max_time_slide_shift``
     - -
     - 100
     - -
   * - ``score_reduction_max``
     - -
     - 5
     - -
   * - ``time_shift_tol``
     - -
     - 0.01
     - -
   * - ``minimal_time_slide_jump``
     - -
     - 0.1
     - -
   * - ``max_zero_lag_delay``
     - -
     - 0.015
     - Maximum delay between detectors within the same timeslide
   * - ``opt_format``
     - -
     - 'new'
     - How we choose the finer grid, changed between O1 and O2 analyses Exposed here to replicate old runs if needed
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
   * - ``outfile_format``
     - -
     - 'new'
     - Flag whether to save the old style (separate npy files for different arrays), or in the new style with a consolidated file per job
   * - ``job_name``
     - -
     - 'coincidence'
     - -
   * - ``n_cores``
     - -
     - 1
     - Number of cores to use for splitting the veto computations
   * - ``n_hours_limit``
     - -
     - 12
     - -
   * - ``epoch_list``
     - -
     - None
     - If known, list of epochs to do coincidence for
   * - ``n_jobs``
     - -
     - 256
     - -
   * - ``mem_limit``
     - -
     - None
     - Memory limit in GB per core requested on cluster
   * - ``debug``
     - -
     - False
     - -
   * - ``submit``
     - -
     - False
     - -
   * - ``overwrite``
     - -
     - False
     - Flag whether to overwrite files (if the base file exists, we preserve it with the suffix "_old"). If int(overwrite) == : 0: Does not touch existing files 1: Redoes the computations and overwrites the existing file 2: Avoids redoing computations and continues the weaker detectors if missing (Note: Only works with outfile_format == "new")
   * - ``cluster``
     - -
     - 'typhon'
     - -
   * - ``exclusive``
     - -
     - False
     - -
   * - ``run``
     - -
     - 'O3a'
     - String identifying the run
   * - ``cver``
     - -
     - 6
     - Cand version
   * - ``rerun``
     - -
     - False
     - Set this flag if rerunning, in which case we append the suffix "_new" to filenames
   * - ``old_cver``
     - -
     - None
     - Old cand run to use to avoid redoing vetoes, if known
   * - ``exclude_nodes``
     - -
     - False
     - Flag to exclude a few nodes from the job (currently only implemented for typhon cluster)

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

   :param dir_path: Path with output files of the triggering
   :param output_dir: Where to place the coincident_files, if known
   :param threshold_chi2:
   :param min_veto_chi2:
   :param max_time_slide_shift:
   :param score_reduction_max:
   :param time_shift_tol:
   :param minimal_time_slide_jump:
   :param max_zero_lag_delay:
       Maximum delay between detectors within the same timeslide
   :param opt_format:
       How we choose the finer grid, changed between O1 and O2 analyses
       Exposed here to replicate old runs if needed
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
   :param outfile_format:
       Flag whether to save the old style (separate npy files for different
       arrays), or in the new style with a consolidated file per job
   :param job_name:
   :param n_cores: Number of cores to use for splitting the veto computations
   :param n_hours_limit:
   :param epoch_list: If known, list of epochs to do coincidence for
   :param n_jobs:
   :param mem_limit: Memory limit in GB per core requested on cluster
   :param debug:
   :param submit:
   :param overwrite:
       Flag whether to overwrite files (if the base file exists, we preserve
       it with the suffix "_old"). If int(overwrite) == :
       0: Does not touch existing files
       1: Redoes the computations and overwrites the existing file
       2: Avoids redoing computations and continues the weaker detectors if
          missing (Note: Only works with outfile_format == "new")
   :param cluster:
   :param exclusive:
   :param run: String identifying the run
   :param cver: Cand version
   :param rerun:
       Set this flag if rerunning, in which case we append the suffix "_new"
       to filenames
   :param old_cver: Old cand run to use to avoid redoing vetoes, if known
   :param exclude_nodes: Flag to exclude a few nodes from the job
                       (currently only implemented for typhon cluster)
   :return:
