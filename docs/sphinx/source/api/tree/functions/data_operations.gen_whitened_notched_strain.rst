data_operations.gen_whitened_notched_strain
===========================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Whiten and notch, modify strain_wt_down in place, and perhaps identify varying lines in mask_freqs

Signature
---------

.. code-block:: python

   def gen_whitened_notched_strain(strain_down, wt_filter_fd_fft, valid_mask, strain_wt_down, fftsize, support_wt, norm_wt, edgefilling = True, edgesafety = 1, notch_wt_filter = False, notch_pars = None, detect_varying_lines = False, **varying_lines_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``strain_down``
     - -
     - -
     - Raw strain data
   * - ``wt_filter_fd_fft``
     - -
     - -
     - FFT of wt_filter in TD
   * - ``valid_mask``
     - -
     - -
     - Boolean mask with zeros where we cannot trust whitened data
   * - ``strain_wt_down``
     - -
     - -
     - Modifies in place
   * - ``fftsize``
     - -
     - -
     - -
   * - ``support_wt``
     - -
     - -
     - Support of filter (TD filter has 2 \\\* support - 1 nonzero coeffs)
   * - ``norm_wt``
     - -
     - -
     - Normalization factor for whitening: (1 / fmax_psd = 2 \\\* dt)\\\*\\\* 0.5
   * - ``edgefilling``
     - -
     - True
     - Flag to fill holes at the edges
   * - ``edgesafety``
     - -
     - 1
     - Safety margin at the edge where we cannot trust the nature of the whitened data (we haven't applied this to the mask when we get here)
   * - ``notch_wt_filter``
     - -
     - False
     - Flag to apply notches to the whitening filter as well, setting it to True avoids biasing the SNR, but creates artifacts at low frequencies (safe to use False as well, since we have PSD drift downstream)
   * - ``notch_pars``
     - -
     - None
     - Parameters for applying the notch_wt_filter
   * - ``detect_varying_lines``
     - -
     - False
     - Identify varying lines
   * - ``\*\*varying_lines_kwargs``
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
     - If detecting varying lines 1. mask_freqs: Boolean mask on freqs with zeros at varying lines 2. amp_ratio: Array with amplitude scaling factors needed to suppress lines above upper threshold

Docstring
---------

.. code-block:: text

   Whiten and notch, modify strain_wt_down in place, and perhaps identify 
   varying lines in mask_freqs
   :param strain_down: Raw strain data
   :param wt_filter_fd_fft: FFT of wt_filter in TD
   :param valid_mask:
       Boolean mask with zeros where we cannot trust whitened data
   :param strain_wt_down: Modifies in place
   :param support_wt: Support of filter (TD filter has 2 * support - 1 nonzero coeffs)
   :param norm_wt: Normalization factor for whitening: (1 / fmax_psd = 2 * dt)** 0.5
   :param detect_varying_lines: Identify varying lines
   :param edgefilling:  Flag to fill holes at the edges
   :param edgesafety:
       Safety margin at the edge where we cannot trust the nature of the
       whitened data (we haven't applied this to the mask when we get here)
   :param notch_wt_filter:
       Flag to apply notches to the whitening filter as well, setting it to
       True avoids biasing the SNR, but creates artifacts at low frequencies
       (safe to use False as well, since we have PSD drift downstream)
   :param notch_pars: Parameters for applying the notch_wt_filter
   :return: If detecting varying lines
       1. mask_freqs: Boolean mask on freqs with zeros at varying lines
       2. amp_ratio: Array with amplitude scaling factors needed to suppress lines
                     above upper threshold
