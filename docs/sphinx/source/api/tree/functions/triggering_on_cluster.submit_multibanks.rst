triggering_on_cluster.submit_multibanks
=======================================

Back to :doc:`Module page <../modules/triggering_on_cluster>`

Summary
-------

Submits multibanks to the cluster

Signature
---------

.. code-block:: python

   def submit_multibanks(tbp_to_use, files_to_submit = None, bank_ids = BBH_KEYS, cluster = 'typhon', observing_run = 'O3a', test_few = None, output_dirs = None, submit = True, save_hole_correction = True, preserve_max_snr = params.DEF_PRESERVE_MAX_SNR, fmax = params.FMAX_OVERLAP, n_cores = None, n_hours_limit = 24, fftlog2size = DEFAULT_FFTLOG2SIZE, njobchunks = 1, trim_empty = False, queue_name = None, env_command = None, use_HM = False, mem_per_cpu = 4, exclude_nodes = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``tbp_to_use``
     - -
     - -
     - Template bank parameter object to use
   * - ``files_to_submit``
     - -
     - None
     - If desired, list of files to run
   * - ``bank_ids``
     - -
     - BBH_KEYS
     - List of bank_ids, keys to dictionaries in template_bank_params....py
   * - ``cluster``
     - -
     - 'typhon'
     - Name of cluster we are submitting jobs to
   * - ``observing_run``
     - -
     - 'O3a'
     - If files_to_submit is None, we use the name of observing run (can be "O1", "O1new", "O2", "O3a", or "O3b")
   * - ``test_few``
     - -
     - None
     - If desired, number of files to submit as a test
   * - ``output_dirs``
     - -
     - None
     - Nested lists of output directories as n_bank_ids x n_subbanks. One can also specify list of particular sub-banks if needed, note that dirnames should end with subbank #. If None, we will generate the names in create_trig_dir_name()
   * - ``submit``
     - -
     - True
     - Flag whether to submit to the cluster
   * - ``save_hole_correction``
     - -
     - True
     - Flag whether to save hole correction, exposed here to rerun files in run "O1"
   * - ``preserve_max_snr``
     - -
     - params.DEF_PRESERVE_MAX_SNR
     - Exposed here since it changed between "O1" and subsequent runs
   * - ``fmax``
     - -
     - params.FMAX_OVERLAP
     - Exposed here since it changes between BBH-0 and the rest of the banks
   * - ``n_cores``
     - -
     - None
     - Number of cores to use for each file, we use sensible values if not provided
   * - ``n_hours_limit``
     - -
     - 24
     - Number of hours to limit the job to. If n_hours_limit<=24, we may get into a faster queue in the cluster
   * - ``fftlog2size``
     - -
     - DEFAULT_FFTLOG2SIZE
     - log_2(fftsize) to use for submission
   * - ``njobchunks``
     - -
     - 1
     - Number of chunks to split the job into, useful if we want to save trigger files in between and restart jobs (useful when n_hours_limit<=24)
   * - ``trim_empty``
     - -
     - False
     - Trim apparently empty files
   * - ``queue_name``
     - -
     - None
     - name of queue (used only in WEXAC submission)
   * - ``env_command``
     - -
     - None
     - If required, pass the command to activate the conda environment If None, will infer from the username if it's in utils
   * - ``use_HM``
     - -
     - False
     - Boolean flag indicating whether we want to use 33 and 44 modes alongside 22
   * - ``mem_per_cpu``
     - -
     - 4
     - Memory per CPU in GB. 4GB/core is the default in Typhon
   * - ``exclude_nodes``
     - -
     - False
     - Flag to exclude nodes from the job submission (if other people start complaining about cluster use)

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

   Submits multibanks to the cluster
   :param tbp_to_use: Template bank parameter object to use
   :param files_to_submit: If desired, list of files to run
   :param bank_ids:
       List of bank_ids, keys to dictionaries in template_bank_params....py
   :param cluster: Name of cluster we are submitting jobs to
   :param observing_run: If files_to_submit is None,
       we use the name of observing run (can be "O1", "O1new", "O2",
                         "O3a", or "O3b")
   :param test_few: If desired, number of files to submit as a test
   :param output_dirs:
       Nested lists of output directories
       as n_bank_ids x n_subbanks.
       One can also specify list of particular sub-banks if needed,
       note that dirnames should end with subbank #.
       If None, we will generate the names in create_trig_dir_name()
   :param submit: Flag whether to submit to the cluster
   :param save_hole_correction:
       Flag whether to save hole correction, exposed here to rerun files
       in run "O1"
   :param preserve_max_snr:
       Exposed here since it changed between "O1" and subsequent runs
   :param fmax:
       Exposed here since it changes between BBH-0 and the rest of the banks
   :param n_cores:
       Number of cores to use for each file, we use sensible values if
       not provided
   :param n_hours_limit:
       Number of hours to limit the job to.
       If n_hours_limit<=24, we may get into a faster queue in the cluster
   :param fftlog2size: log_2(fftsize) to use for submission
   :param njobchunks:
       Number of chunks to split the job into, useful if we want to save
       trigger files in between and restart jobs
       (useful when n_hours_limit<=24)
   :param trim_empty: Trim apparently empty files
   :param queue_name: name of queue (used only in WEXAC submission)
   :param env_command:
           If required, pass the command to activate the conda environment
           If None, will infer from the username if it's in utils
   :param use_HM:
           Boolean flag indicating whether we want to use 33 and 44 modes alongside 22
   :param mem_per_cpu: Memory per CPU in GB. 4GB/core is the default in Typhon
   :param exclude_nodes: Flag to exclude nodes from the job submission
                       (if other people start complaining about cluster use)
   :return:
