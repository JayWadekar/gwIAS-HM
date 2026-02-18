triggering_on_cluster.submit_files_typhon
=========================================

Back to :doc:`Module page <../modules/triggering_on_cluster>`

Summary
-------

No docstring summary available.

Signature
---------

.. code-block:: python

   def submit_files_typhon(fnames, mem_per_cpu = 4, n_hours_limit = 24, output_dir = None, template_conf = None, delta_calpha = 0.5, template_safety = 1.05, remove_nonphysical = True, force_zero = True, base_threshold_chi2 = 16, threshold_chi2 = 25, save_hole_correction = True, preserve_max_snr = params.DEF_PRESERVE_MAX_SNR, fmax = params.FMAX_OVERLAP, prog_path = DEFAULT_PROGPATH, job_name = DEFAULT_JOBNAME, ncores = 24, fftlog2size = 20, n_waveforms_per_core = -1, njobchunks = 1, run_str = 'O2', check_completion = True, submit = False, env_command = None, use_HM = False, exclude_nodes = False, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fnames``
     - -
     - -
     - -
   * - ``mem_per_cpu``
     - -
     - 4
     - -
   * - ``n_hours_limit``
     - -
     - 24
     - -
   * - ``output_dir``
     - -
     - None
     - -
   * - ``template_conf``
     - -
     - None
     - -
   * - ``delta_calpha``
     - -
     - 0.5
     - -
   * - ``template_safety``
     - -
     - 1.05
     - -
   * - ``remove_nonphysical``
     - -
     - True
     - -
   * - ``force_zero``
     - -
     - True
     - -
   * - ``base_threshold_chi2``
     - -
     - 16
     - -
   * - ``threshold_chi2``
     - -
     - 25
     - -
   * - ``save_hole_correction``
     - -
     - True
     - -
   * - ``preserve_max_snr``
     - -
     - params.DEF_PRESERVE_MAX_SNR
     - -
   * - ``fmax``
     - -
     - params.FMAX_OVERLAP
     - -
   * - ``prog_path``
     - -
     - DEFAULT_PROGPATH
     - -
   * - ``job_name``
     - -
     - DEFAULT_JOBNAME
     - -
   * - ``ncores``
     - -
     - 24
     - -
   * - ``fftlog2size``
     - -
     - 20
     - -
   * - ``n_waveforms_per_core``
     - -
     - -1
     - -
   * - ``njobchunks``
     - -
     - 1
     - -
   * - ``run_str``
     - -
     - 'O2'
     - -
   * - ``check_completion``
     - -
     - True
     - -
   * - ``submit``
     - -
     - False
     - -
   * - ``env_command``
     - -
     - None
     - -
   * - ``use_HM``
     - -
     - False
     - -
   * - ``exclude_nodes``
     - -
     - False
     - -
   * - ``\*\*kwargs``
     - -
     - -
     - -

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

No docstring text available.
