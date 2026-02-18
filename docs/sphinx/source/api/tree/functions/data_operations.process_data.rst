data_operations.process_data
============================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Preferably send in 2\\\*\\\*N samples

Signature
---------

.. code-block:: python

   def process_data(time, strain, mask, asd_func_data, do_signal_processing = True, sigma_clipping_threshold = None, sine_gaussian_intervals = None, sine_gaussian_thresholds = None, bandlim_transient_intervals = None, bandlim_power_thresholds = None, excess_power_intervals = None, excess_power_thresholds = None, erase_bands = False, freqs_in = None, crude_line_mask = None, loud_line_mask = None, freqs_to_notch = None, notch_wt_filter = False, renorm_wt = True, times_to_fill = None, times_to_save = None, fmax = params.FMAX_OVERLAP, notch_format = 'old', taper_wt_filter = False, taper_fraction = 0.5, min_filt_trunc_time = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``time``
     - -
     - -
     - Array of times (s)
   * - ``strain``
     - -
     - -
     - Array of strain data
   * - ``mask``
     - -
     - -
     - Quality mask on strain data at 1 Hz
   * - ``asd_func_data``
     - -
     - -
     - Function returning ASDs (1/sqrt{Hz}) given frequencies (Hz)
   * - ``do_signal_processing``
     - -
     - True
     - Flag indicating whether to do signal processing to identify glitches and fill holes
   * - ``sigma_clipping_threshold``
     - -
     - None
     - Threshold for sigma clipping calculated from waveform. If None, clipping depends on params.NPERFILE
   * - ``sine_gaussian_intervals``
     - -
     - None
     - Frequency bands within which we look for Sine-Gaussian noise [central frequency, df = (upper - lower frequency)] Hz
   * - ``sine_gaussian_thresholds``
     - -
     - None
     - Amplitude thresholds for sine gaussian transients computed from waveforns. If None, detection depends on params.NPERFILE
   * - ``bandlim_transient_intervals``
     - -
     - None
     - Array with set of time-interval-frequency intervals (s, Hz) [[dt_i, [f_i_min, f_i_max]],...]
   * - ``bandlim_power_thresholds``
     - -
     - None
     - Thresholds for bandlimited transient detection computed from waveforms. If None, detection depends on params.NPERFILE
   * - ``excess_power_intervals``
     - -
     - None
     - Array with timescales to look for excess power on (s)
   * - ``excess_power_thresholds``
     - -
     - None
     - Thresholds for excess power detection computed from waveforms. If None, detection depends on params.NPERFILE
   * - ``erase_bands``
     - -
     - False
     - Flag to erase bands in time-frequency space
   * - ``freqs_in``
     - -
     - None
     - Array with frequencies on which we looked for lines (optional)
   * - ``crude_line_mask``
     - -
     - None
     - Boolean array with zeros marking crudely identified lines (optional)
   * - ``loud_line_mask``
     - -
     - None
     - Boolean array with zeros marking identified loud lines, that will be notched (optional)
   * - ``freqs_to_notch``
     - -
     - None
     - If desired, list of frequency ranges to notch [[f_min_i, f_max_i]]
   * - ``notch_wt_filter``
     - -
     - False
     - Flag to apply notches to the whitening filter as well, setting it to True avoids biasing the SNR, but creates artifacts at low frequencies (safe to use False as well, since we have PSD drift downstream)
   * - ``renorm_wt``
     - -
     - True
     - Flag whether to scale the whitened data to have unit variance after highpass (if True, we scale by some amount to account for the fraction of the band lost)
   * - ``times_to_fill``
     - -
     - None
     - If desired, list of time ranges to create holes in [[t_min, t_max], ...]
   * - ``times_to_save``
     - -
     - None
     - If desired, list of time ranges not to create holes in [[t_min, t_max], ...]
   * - ``fmax``
     - -
     - params.FMAX_OVERLAP
     - Maximum frequency involved in the analysis
   * - ``notch_format``
     - -
     - 'old'
     - Flag to pick the format for notching. "new" tries to center notches on the finer grid if available
   * - ``taper_wt_filter``
     - -
     - False
     - Flag whether to taper the time domain response of the whitening filter with a Tukey window
   * - ``taper_fraction``
     - -
     - 0.5
     - Fraction of response to taper with a Tukey window, if applicable (0 is boxcar, 1 is Hann)
   * - ``min_filt_trunc_time``
     - -
     - 1
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
     - 1. Array of downsampled times 2. Downsampled and whitened strain data 3. Boolean mask with zeros where data has been zeroed/inpainted 4. Boolean mask with zeros where we cannot trust whitened data 5. Fine grequency grid on which we identify lines 6. Mask on the fine frequency grid with zeros at varying lines 7. n_glitch_tests x ntimes boolean array with 0 where test fired 8. Whitening filter in time domain (array of length fftsize, with weight at the ends) 9. Support of whitening filter (TD filter has 2 \\\* support - 1 nonzero coeffs) 10. Normalization factor for whitening filter (to whiten the data, convolve with norm \\\* whitening_filter_fd) 11. Factor to multiply the whitened strain with to get each entry to be N(0,1) (differs from 1 due to bandpass, the value = 1 = incorrect if notch_wt_filter is True) 12. Freqs x times boolean mask with zeros at turbulent bands

Docstring
---------

.. code-block:: text

   Preferably send in 2**N samples
   :param time: Array of times (s)
   :param strain: Array of strain data
   :param mask: Quality mask on strain data at 1 Hz
   :param asd_func_data:
       Function returning ASDs (1/sqrt{Hz}) given frequencies (Hz)
   :param do_signal_processing:
       Flag indicating whether to do signal processing to identify glitches
       and fill holes
   :param sigma_clipping_threshold:
       Threshold for sigma clipping calculated from waveform. If None, clipping
       depends on params.NPERFILE
   :param sine_gaussian_intervals:
       Frequency bands within which we look for Sine-Gaussian noise
       [central frequency, df = (upper - lower frequency)] Hz
   :param sine_gaussian_thresholds:
       Amplitude thresholds for sine gaussian transients computed from
       waveforns. If None, detection depends on params.NPERFILE
   :param bandlim_transient_intervals:
       Array with set of time-interval-frequency intervals (s, Hz)
       [[dt_i, [f_i_min, f_i_max]],...]
   :param bandlim_power_thresholds:
       Thresholds for bandlimited transient detection computed from waveforms.
       If None, detection depends on params.NPERFILE
   :param excess_power_intervals:
       Array with timescales to look for excess power on (s)
   :param excess_power_thresholds:
       Thresholds for excess power detection computed from waveforms. If None,
       detection depends on params.NPERFILE
   :param erase_bands: Flag to erase bands in time-frequency space
   :param freqs_in:
       Array with frequencies on which we looked for lines (optional)
   :param crude_line_mask:
       Boolean array with zeros marking crudely identified lines (optional)
   :param loud_line_mask:
       Boolean array with zeros marking identified loud lines, that will be
       notched (optional)
   :param freqs_to_notch:
       If desired, list of frequency ranges to notch [[f_min_i, f_max_i]]
   :param notch_wt_filter:
       Flag to apply notches to the whitening filter as well, setting it to
       True avoids biasing the SNR, but creates artifacts at low frequencies
       (safe to use False as well, since we have PSD drift downstream)
   :param renorm_wt:
       Flag whether to scale the whitened data to have unit variance after
       highpass (if True, we scale by some amount to account for the fraction
       of the band lost)
   :param times_to_fill:
       If desired, list of time ranges to create holes in [[t_min, t_max], ...]
   :param times_to_save:
       If desired, list of time ranges not to create holes in
       [[t_min, t_max], ...]
   :param fmax: Maximum frequency involved in the analysis
   :param notch_format:
       Flag to pick the format for notching. "new" tries to center notches on
       the finer grid if available
   :param taper_wt_filter:
       Flag whether to taper the time domain response of the whitening filter
       with a Tukey window
   :param taper_fraction:
       Fraction of response to taper with a Tukey window, if applicable
       (0 is boxcar, 1 is Hann)
   :return: 1. Array of downsampled times
            2. Downsampled and whitened strain data
            3. Boolean mask with zeros where data has been zeroed/inpainted
            4. Boolean mask with zeros where we cannot trust whitened data
            5. Fine grequency grid on which we identify lines
            6. Mask on the fine frequency grid with zeros at varying lines
            7. n_glitch_tests x ntimes boolean array with 0 where test fired
            8. Whitening filter in time domain (array of length fftsize, with
               weight at the ends)
            9. Support of whitening filter (TD filter has
               2 * support - 1 nonzero coeffs)
            10. Normalization factor for whitening filter (to whiten the data,
               convolve with norm * whitening_filter_fd)
            11. Factor to multiply the whitened strain with to get each entry
                to be N(0,1) (differs from 1 due to bandpass, the value = 1 =
                incorrect if notch_wt_filter is True)
            12. Freqs x times boolean mask with zeros at turbulent bands
