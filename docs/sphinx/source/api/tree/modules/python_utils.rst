python_utils
============

Back to :doc:`API tree index <../index>`

Purpose
-------

Numeric and Python utility helpers shared across modules.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`abbar <../functions/python_utils.abbar>`
     - return c1 \\\* np.conjugate(c2) with numba vectorization
   * - :doc:`abs2 <../functions/python_utils.abs2>`
     - x.real^2 + x.imag^2
   * - :doc:`abs2sum <../functions/python_utils.abs2sum>`
     - sum x.real^2 + x.imag^2 along last axis
   * - :doc:`archive_paths <../functions/python_utils.archive_paths>`
     - No docstring summary available.
   * - :doc:`argmax_lastax <../functions/python_utils.argmax_lastax>`
     - get axis=-1 (innermost \\\`column') index of max in arr
   * - :doc:`argmaxnd <../functions/python_utils.argmaxnd>`
     - get unraveled (tuple) index of maximal element in arr
   * - :doc:`check_equal <../functions/python_utils.check_equal>`
     - recursive x == y that handles ragged mixed-type containers
   * - :doc:`flat_in_log <../functions/python_utils.flat_in_log>`
     - get n samples from log-uniform distribution between vmin and vmax
   * - :doc:`fmt <../functions/python_utils.fmt>`
     - formatting number as string
   * - :doc:`import_matplotlib <../functions/python_utils.import_matplotlib>`
     - Avoid annoying errors due to loading matplotlib on the server
   * - :doc:`invert_dict <../functions/python_utils.invert_dict>`
     - return dictionary with inverted key-value pairs
   * - :doc:`is_numpy_float <../functions/python_utils.is_numpy_float>`
     - check if type(x) is one of numpy's float types
   * - :doc:`is_numpy_int <../functions/python_utils.is_numpy_int>`
     - check if type(x) is one of numpy's integer types
   * - :doc:`load_module <../functions/python_utils.load_module>`
     - load python module by name (string)
   * - :doc:`load_symmetrix_matrix <../functions/python_utils.load_symmetrix_matrix>`
     - load n x n symmetric matrix from 1d array formatted as in store_symmetric_matrix()
   * - :doc:`merge_dicts_safely <../functions/python_utils.merge_dicts_safely>`
     - merge multiple dictionaries into one, accepting repeated keys if values are consistent, otherwise raise ValueError
   * - :doc:`next_power <../functions/python_utils.next_power>`
     - get next power of 2 following n
   * - :doc:`nfft_of_rfft <../functions/python_utils.nfft_of_rfft>`
     - ASSUMES nfft even <=> len(rfft_arr) odd
   * - :doc:`nfft_of_rfftlen <../functions/python_utils.nfft_of_rfftlen>`
     - ASSUMES nfft even <=> rfftlen odd
   * - :doc:`npy_append_cols <../functions/python_utils.npy_append_cols>`
     - append columns to array from .npy file at path infile if outfile_new given, write the result there; otherwise rewrite infile_npy with appended columns
   * - :doc:`npy_append_rows <../functions/python_utils.npy_append_rows>`
     - append rows to array from .npy file at path infile if outfile_new given, write the result there; otherwise rewrite infile_npy with appended rows
   * - :doc:`printarr <../functions/python_utils.printarr>`
     - wrapper for np.array2string printing
   * - :doc:`rand_azim <../functions/python_utils.rand_azim>`
     - get nangles (int or tuple) samples ~ U(0, 2\\\*np.pi)
   * - :doc:`rand_cos <../functions/python_utils.rand_cos>`
     - get nangles (int or tuple) samples ~ U(-1, 1)
   * - :doc:`rand_polar <../functions/python_utils.rand_polar>`
     - get nangles (int or tuple) samples ~ np.arccos(U(-1, 1))
   * - :doc:`rfftlen_of_nfft <../functions/python_utils.rfftlen_of_nfft>`
     - ASSUMES nfft even <=> len(rfft_arr) odd
   * - :doc:`store_symmetrix_matrix <../functions/python_utils.store_symmetrix_matrix>`
     - store n x n symmetric matrix as 1d array of length n\\\*(n+1)/2
   * - :doc:`tukwin_back <../functions/python_utils.tukwin_back>`
     - back taper of tukey window (last half of hann)
   * - :doc:`tukwin_bandpass <../functions/python_utils.tukwin_bandpass>`
     - taper_width is frequency interval length in Hz to be tapered at each end f_nyq & rfft_len are the nyquist frequency and length of the rfftfreq array
   * - :doc:`tukwin_front <../functions/python_utils.tukwin_front>`
     - front taper of tukey window (first half of hann)
   * - :doc:`tukwin_npts <../functions/python_utils.tukwin_npts>`
     - get tukey window based on numbers of total and windowed points
   * - :doc:`zeropad_end <../functions/python_utils.zeropad_end>`
     - pad arr axis=-1 with zeros at (right/back) end to length pad_to_N if arr.shape[-1] < pad_to_N, raise ValueError
   * - :doc:`zip_to_array <../functions/python_utils.zip_to_array>`
     - O(10^3)x faster np.array([row for row in np.broadcast(\\\*args)]) usable when args are all at most 1d.

No public classes were found.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../functions/python_utils.abbar
   ../functions/python_utils.abs2
   ../functions/python_utils.abs2sum
   ../functions/python_utils.archive_paths
   ../functions/python_utils.argmax_lastax
   ../functions/python_utils.argmaxnd
   ../functions/python_utils.check_equal
   ../functions/python_utils.flat_in_log
   ../functions/python_utils.fmt
   ../functions/python_utils.import_matplotlib
   ../functions/python_utils.invert_dict
   ../functions/python_utils.is_numpy_float
   ../functions/python_utils.is_numpy_int
   ../functions/python_utils.load_module
   ../functions/python_utils.load_symmetrix_matrix
   ../functions/python_utils.merge_dicts_safely
   ../functions/python_utils.next_power
   ../functions/python_utils.nfft_of_rfft
   ../functions/python_utils.nfft_of_rfftlen
   ../functions/python_utils.npy_append_cols
   ../functions/python_utils.npy_append_rows
   ../functions/python_utils.printarr
   ../functions/python_utils.rand_azim
   ../functions/python_utils.rand_cos
   ../functions/python_utils.rand_polar
   ../functions/python_utils.rfftlen_of_nfft
   ../functions/python_utils.store_symmetrix_matrix
   ../functions/python_utils.tukwin_back
   ../functions/python_utils.tukwin_bandpass
   ../functions/python_utils.tukwin_front
   ../functions/python_utils.tukwin_npts
   ../functions/python_utils.zeropad_end
   ../functions/python_utils.zip_to_array
