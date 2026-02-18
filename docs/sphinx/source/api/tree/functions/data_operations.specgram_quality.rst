data_operations.specgram_quality
================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Computes specgram of whitened data, along with list of bad time and frequency channels in the specgram (bad time channels are windows that overlap with zeros in mask, and bad frequency channels are varying lines or those beyond the analysis range)

Signature
---------

.. code-block:: python

   def specgram_quality(strain_wt, qmask, valid_mask, dt, interval, edgesafety = 1, nfire = params.NPERFILE, freqs_lines = None, mask_freqs_in = None, fmax = params.FMAX_OVERLAP, **spec_kwargs)

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
     - Boolean mask with zeros at holes in unwhitened data (we have already widened holes due to the highpass filter)
   * - ``valid_mask``
     - -
     - -
     - Boolean mask with zeros where we cannot trust whitened data
   * - ``dt``
     - -
     - -
     - Time interval between successive elements of strain_wt (s)
   * - ``interval``
     - -
     - -
     - Time interval for FFTs (s, sets resolution of lines)
   * - ``edgesafety``
     - -
     - 1
     - Safety margin at the edge where we cannot trust the nature of the whitened data (we haven't applied this to the mask when we get here)
   * - ``nfire``
     - -
     - params.NPERFILE
     - Number of times line-detector fires per perfect file
   * - ``freqs_lines``
     - -
     - None
     - Array with frequencies on which we previously detected lines (optional)
   * - ``mask_freqs_in``
     - -
     - None
     - Boolean mask on freqs_lines with zeros at previously detected lines/varying lines (optional)
   * - ``fmax``
     - -
     - params.FMAX_OVERLAP
     - Maximum frequency involved in the analysis
   * - ``\*\*spec_kwargs``
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
     - 1. Array of length n_freqs with frequencies (Hz) 2. Array of length n_times with times (s) 3. n_freqs x n_times array with specgram (two-sided psd in Hz^-1) 4. Boolean array of length n_freqs with zeros at candidate varying line channels 5. Boolean array of length n_times with zeros where FFTs see holes/invalid data 6. Array with amplitude scaling factors needed to suppress lines above upper threshold

Docstring
---------

.. code-block:: text

   Computes specgram of whitened data, along with list of bad time and
   frequency channels in the specgram (bad time channels are windows that
   overlap with zeros in mask, and bad frequency channels are varying lines or
   those beyond the analysis range)
   :param strain_wt: Whitened strain data
   :param qmask: Boolean mask with zeros at holes in unwhitened data (we have
                 already widened holes due to the highpass filter)
   :param valid_mask:
       Boolean mask with zeros where we cannot trust whitened data
   :param dt: Time interval between successive elements of strain_wt (s)
   :param interval: Time interval for FFTs (s, sets resolution of lines)
   :param edgesafety:
       Safety margin at the edge where we cannot trust the nature of the
       whitened data (we haven't applied this to the mask when we get here)
   :param nfire: Number of times line-detector fires per perfect file
   :param freqs_lines:
       Array with frequencies on which we previously detected lines (optional)
   :param mask_freqs_in:
       Boolean mask on freqs_lines with zeros at previously detected
       lines/varying lines (optional)
   :param fmax: Maximum frequency involved in the analysis
   :return: 1. Array of length n_freqs with frequencies (Hz)
            2. Array of length n_times with times (s)
            3. n_freqs x n_times array with specgram (two-sided psd in Hz^-1)
            4. Boolean array of length n_freqs with zeros at candidate
               varying line channels
            5. Boolean array of length n_times with zeros where FFTs see
               holes/invalid data
            6. Array with amplitude scaling factors needed to suppress lines
               above upper threshold
