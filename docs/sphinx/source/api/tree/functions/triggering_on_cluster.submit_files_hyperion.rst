triggering_on_cluster.submit_files_hyperion
===========================================

Back to :doc:`Module page <../modules/triggering_on_cluster>`

Summary
-------

Hyperion doesn't have as much memory, so suggest using exclusive

Signature
---------

.. code-block:: python

   def submit_files_hyperion(fnames, output_dir = None, template_conf = None, delta_calpha = 0.7, template_safety = 1.1, remove_nonphysical = True, force_zero = True, base_threshold_chi2 = 16, threshold_chi2 = 25, save_hole_correction = True, preserve_max_snr = params.DEF_PRESERVE_MAX_SNR, fmax = params.FMAX_OVERLAP, prog_path = DEFAULT_PROGPATH, job_name = DEFAULT_JOBNAME, ncores = 8, fftlog2size = 20, n_hours_limit = 8, n_waveforms_per_core = -1, njobchunks = 1, run_str = 'O2', check_completion = True, submit = False, exclusive = False, **kwargs)

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
     - 0.7
     - -
   * - ``template_safety``
     - -
     - 1.1
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
     - 8
     - -
   * - ``fftlog2size``
     - -
     - 20
     - -
   * - ``n_hours_limit``
     - -
     - 8
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
   * - ``exclusive``
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

.. code-block:: text

   Hyperion doesn't have as much memory, so suggest using exclusive
