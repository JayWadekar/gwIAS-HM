template_bank_generator_HM.TemplateBank.test_effectualness
==========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Computes the mismatches between original astrophysical waveforms and the bank's representation of them. Used to measure losses in sensitivity

Signature
---------

.. code-block:: python

   def test_effectualness(self, wfs_fd, delta_calpha, fs_in = None, HM = True, wfs_22_fd = None, n_sinc_interp = 2, highpass = True, truncate = True, coeff_grid = None, do_optimization = True, fudge = params.TEMPLATE_SAFETY, force_zero = True, ret_overlaps_guess = False, search_full_bank = False, calpha_search_radius = 5)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wfs_fd``
     - -
     - -
     - n_wf x fs_in array with LOG of frequency domain waveforms on fs_in, If using 2,2 modes only, ensure that the phases are unwrapped so that we can linearly interpolate If using 2,2 + higher modes, interpolation won't be accurate so provide them on self.fs_fft
   * - ``delta_calpha``
     - -
     - -
     - Resolution of calphas used for coeff_grid (the finer grid will have 0.5 \\\* delta_calpha)
   * - ``fs_in``
     - -
     - None
     - Frequency grid for wfs_fd (None = self.fs_fft)
   * - ``HM``
     - -
     - True
     - Flag indicating whether we are using higher mode template bank or 22 only
   * - ``wfs_22_fd``
     - -
     - None
     - If HM is true, provide a separate 22-only version of the full waveform. These 22 wfs are only used to find the best-fit calphas (ensure that the phases are unwrapped so that we can linearly interpolate) Format: n_wf x fs_in array with LOG of frequency domain waveforms on fs_in
   * - ``n_sinc_interp``
     - -
     - 2
     - Number of times to sinc-interpolate the overlaps
   * - ``highpass``
     - -
     - True
     - Flag indicating whether to high-pass filter the waveforms
   * - ``truncate``
     - -
     - True
     - -
   * - ``coeff_grid``
     - -
     - None
     - n_templates x n_dims array with the coefficients of the templates that are in the bank
   * - ``do_optimization``
     - -
     - True
     - -
   * - ``fudge``
     - -
     - params.TEMPLATE_SAFETY
     - -
   * - ``force_zero``
     - -
     - True
     - -
   * - ``ret_overlaps_guess``
     - -
     - False
     - Flag indicating whether to include the guess in the return (guess = sum_i \\\|delta_calpha_i\\\*\\\*2\\\|). Only used for debugging.
   * - ``search_full_bank``
     - -
     - False
     - Flag indicating whether to search the full bank. If false (default), we first find the best-fit calpha template and only take the overlap with that. Otherwise, we take overlaps with all templates in bank (more expensive). Internally, we only implement this case when overlap_threshold<0.96. Suggestion: Better to not use this flag with multiprocessing as it consumes a lot of memory.
   * - ``calpha_search_radius``
     - -
     - 5
     - This will be updated later by Mark. #TODO_Mark Applicable if search_full_bank is true. We only search for templates within this radius of the best-fit calpha. If None, we search the full bank.

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

   Computes the mismatches between original astrophysical waveforms and
   the bank's representation of them. Used to measure losses in sensitivity
   :param wfs_fd:
       n_wf x fs_in array with LOG of frequency domain waveforms on fs_in,
       If using 2,2 modes only, ensure that the phases are unwrapped so
       that we can linearly interpolate
       If using 2,2 + higher modes, interpolation won't be accurate so
       provide them on self.fs_fft
   :param fs_in: Frequency grid for wfs_fd (None = self.fs_fft)
   :param HM: Flag indicating whether we are using higher mode template
              bank or 22 only
   :param wfs_22_fd:
       If HM is true, provide a separate 22-only version of the full waveform.
       These 22 wfs are only used to find the best-fit calphas (ensure that the
        phases are unwrapped so that we can linearly interpolate)
       Format: n_wf x fs_in array with LOG of frequency domain waveforms on fs_in
   :param delta_calpha:
       Resolution of calphas used for coeff_grid
       (the finer grid will have 0.5 * delta_calpha)
   :param coeff_grid:
       n_templates x n_dims array with the coefficients of the templates
       that are in the bank
   :param n_sinc_interp: Number of times to sinc-interpolate the overlaps
   :param highpass:
       Flag indicating whether to high-pass filter the waveforms
   :param ret_overlaps_guess: Flag indicating whether to include the guess
            in the return (guess = sum_i |delta_calpha_i**2|).
            Only used for debugging.
   :param search_full_bank: Flag indicating whether to search the full bank.
       If false (default), we first find the best-fit calpha template and
       only take the overlap with that. Otherwise, we take overlaps with
       all templates in bank (more expensive). Internally, we only
       implement this case when overlap_threshold<0.96.
       Suggestion: Better to not use this flag with multiprocessing 
       as it consumes a lot of memory.
   :param calpha_search_radius:
       This will be updated later by Mark. #TODO_Mark
       Applicable if search_full_bank is true. We only search for templates
         within this radius of the best-fit calpha. If None, we search the full bank.
   :return if n_sinc_interp=0,
           Array of size n_wf with maximum overlaps at current resolution
           else:
           Array of size n_wf x 2
           Maximum overlaps at current resolution
           Maximum sinc interpolated overlaps, without angle information (use this as default)
