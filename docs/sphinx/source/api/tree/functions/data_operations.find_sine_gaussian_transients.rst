data_operations.find_sine_gaussian_transients
=============================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Finds sine-gaussian transients in data

Signature
---------

.. code-block:: python

   def find_sine_gaussian_transients(strain_wt, qmask, valid_mask, dt, clipwidth, freqs_lines, mask_freqs, sine_gaussian_intervals = None, sine_gaussian_thresholds = None, edgesafety = 1, fftsize = params.DEF_FFTSIZE, zero_mask = True, nfire = params.NPERFILE, renorm_wt = True, mask_save = None, verbose = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``strain_wt``
     - -
     - -
     - Whitened strain data
   * - ``qmask``
     - -
     - -
     - Boolean mask with zeros at holes in unwhitened data
   * - ``valid_mask``
     - -
     - -
     - Boolean mask with zeros where we cannot trust whitened data
   * - ``dt``
     - -
     - -
     - Time interval between successive elements of data (s)
   * - ``clipwidth``
     - -
     - -
     - Half-width of window around outliers to zero (in s)
   * - ``freqs_lines``
     - -
     - -
     - Array with frequencies over which we detected lines
   * - ``mask_freqs``
     - -
     - -
     - Boolean mask on freqs with zeros at varying lines
   * - ``sine_gaussian_intervals``
     - -
     - None
     - Frequency bands within which we look for Sine-Gaussian noise [central frequency, df = (upper - lower frequency)] Hz
   * - ``sine_gaussian_thresholds``
     - -
     - None
     - Amplitude thresholds for sine gaussian transients computed from waveforms. If None, detection depends on params.NPERFILE
   * - ``edgesafety``
     - -
     - 1
     - Safety margin at the edge where we cannot trust the nature of the whitened data (we haven't applied this to the mask when we get here)
   * - ``fftsize``
     - -
     - params.DEF_FFTSIZE
     - FFTsize for overlap-save
   * - ``zero_mask``
     - -
     - True
     - Flag indicating whether to zero the mask at glitches. Turn off for vetoing
   * - ``nfire``
     - -
     - params.NPERFILE
     - Number of times glitch detector fires per perfect file
   * - ``renorm_wt``
     - -
     - True
     - Flag whether we scaled the whitened data to have unit variance after highpass
   * - ``mask_save``
     - -
     - None
     - If desired, boolean mask on strain_wt with zeros at data to avoid
   * - ``verbose``
     - -
     - True
     - If true, prints details about sine-Gaussian check

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Zeros qmask in place, and returns 1. List of indices that jumped 2. Masks with zero where we zeroed the data, for each interval 3. String with details of the test

Docstring
---------

.. code-block:: text

   Finds sine-gaussian transients in data
   :param strain_wt: Whitened strain data
   :param qmask: Boolean mask with zeros at holes in unwhitened data
   :param valid_mask:
       Boolean mask with zeros where we cannot trust whitened data
   :param dt: Time interval between successive elements of data (s)
   :param clipwidth: Half-width of window around outliers to zero (in s)
   :param freqs_lines: Array with frequencies over which we detected lines
   :param mask_freqs: Boolean mask on freqs with zeros at varying lines
   :param sine_gaussian_intervals:
       Frequency bands within which we look for Sine-Gaussian noise
       [central frequency, df = (upper - lower frequency)] Hz
   :param sine_gaussian_thresholds:
       Amplitude thresholds for sine gaussian transients computed from
       waveforms. If None, detection depends on params.NPERFILE
   :param edgesafety:
       Safety margin at the edge where we cannot trust the nature of the
       whitened data (we haven't applied this to the mask when we get here)
   :param fftsize: FFTsize for overlap-save
   :param zero_mask:
       Flag indicating whether to zero the mask at glitches. Turn off for
       vetoing
   :param nfire: Number of times glitch detector fires per perfect file
   :param renorm_wt:
       Flag whether we scaled the whitened data to have unit variance after
       highpass
   :param mask_save:
       If desired, boolean mask on strain_wt with zeros at data to avoid
   :param verbose: If true, prints details about sine-Gaussian check
   :return: Zeros qmask in place, and returns
            1. List of indices that jumped
            2. Masks with zero where we zeroed the data, for each interval
            3. String with details of the test
