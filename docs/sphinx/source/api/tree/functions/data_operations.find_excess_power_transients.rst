data_operations.find_excess_power_transients
============================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Detects bandlimited excess power transients

Signature
---------

.. code-block:: python

   def find_excess_power_transients(strain_wt, qmask, valid_mask, dt, excess_power_intervals = None, excess_power_thresholds = None, edgesafety = 1, freqs_lines = None, mask_freqs = None, zero_mask = True, nfire = params.NPERFILE, fmax = params.FMAX_OVERLAP, mask_save = None, verbose = True)

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
   * - ``excess_power_intervals``
     - -
     - None
     - Array with set of time-interval-frequency intervals (s, Hz) [[dt_i, [f_i_min, f_i_max]],...]. Pass f_min = 0, f_max = np.inf for total power
   * - ``excess_power_thresholds``
     - -
     - None
     - Thresholds for excess power computed from waveforms. If None, detection depends on params.NPERFILE. The thresholds correspond to squared sum of bandpassed data
   * - ``edgesafety``
     - -
     - 1
     - Safety margin at the edge where we cannot trust the nature of the whitened data (we haven't applied this to the mask when we get here)
   * - ``freqs_lines``
     - -
     - None
     - Array with frequencies on which we previously detected lines (optional)
   * - ``mask_freqs``
     - -
     - None
     - Boolean mask on freqs_lines with zeros at previously detected varying lines (optional)
   * - ``zero_mask``
     - -
     - True
     - Flag indicating whether to zero the mask at glitches. Turn off for vetoing
   * - ``nfire``
     - -
     - params.NPERFILE
     - Number of times glitch detector fires per perfect file
   * - ``fmax``
     - -
     - params.FMAX_OVERLAP
     - Maximum frequency involved in the analysis
   * - ``mask_save``
     - -
     - None
     - If desired, boolean mask on strain_wt with zeros at data to avoid
   * - ``verbose``
     - -
     - True
     - If true, would print details on power detector

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

   Detects bandlimited excess power transients
   :param strain_wt: Whitened strain data
   :param qmask: Boolean mask with zeros at holes in unwhitened data
   :param valid_mask:
       Boolean mask with zeros where we cannot trust whitened data
   :param dt: Time interval between successive elements of data (s)
   :param excess_power_intervals:
       Array with set of time-interval-frequency intervals (s, Hz)
       [[dt_i, [f_i_min, f_i_max]],...]. Pass f_min = 0, f_max = np.inf for
       total power
   :param excess_power_thresholds:
           Thresholds for excess power computed from waveforms. If None,
           detection depends on params.NPERFILE. The thresholds correspond
           to squared sum of bandpassed data
   :param edgesafety:
       Safety margin at the edge where we cannot trust the nature of the
       whitened data (we haven't applied this to the mask when we get here)
   :param freqs_lines:
       Array with frequencies on which we previously detected lines (optional)
   :param mask_freqs:
       Boolean mask on freqs_lines with zeros at previously detected varying
       lines (optional)
   :param zero_mask:
       Flag indicating whether to zero the mask at glitches. Turn off for
       vetoing
   :param nfire: Number of times glitch detector fires per perfect file
   :param fmax: Maximum frequency involved in the analysis
   :param mask_save:
       If desired, boolean mask on strain_wt with zeros at data to avoid
   :param verbose: If true, would print details on power detector
   :return: Zeros qmask in place, and returns
            1. List of indices that jumped
            2. Masks with zero where we zeroed the data, for each interval
            3. String with details of the test
