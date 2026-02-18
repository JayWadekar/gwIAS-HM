triggers_single_detector_HM.TriggerList.prepare_summary_stats
=============================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Defines summary statistics for convolution with relative waveforms The statistics correspond to convolution with the calpha waveform, without any subgrid shifts

Signature
---------

.. code-block:: python

   def prepare_summary_stats(self, trigger = None, location = None, relative_freq_bins = None, dcalphas = None, dt = params.DT_OPT, delta = 0.1, subset_defined = False, relative_binning = True, zero_pad = True, calc_mode_covariance = True, use_HM = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - None
     - Trigger in the form of a row of a processed clist
   * - ``location``
     - -
     - None
     - Tuple of length 2 with (linear-free time, calphas), used if trigger is not given
   * - ``relative_freq_bins``
     - -
     - None
     - Array with bin edges for frequency interpolation
   * - ``dcalphas``
     - -
     - None
     - Array with calpha spacings of finer grid to optimize
   * - ``dt``
     - -
     - params.DT_OPT
     - Shift in time (s) to allow
   * - ``delta``
     - -
     - 0.1
     - Phase allowed to accumulate within each bin
   * - ``subset_defined``
     - -
     - False
     - Flag indicating if we already defined the required subset of data, used to save on FFTs
   * - ``relative_binning``
     - -
     - True
     - Flag indicating whether we are using relative binning
   * - ``zero_pad``
     - -
     - True
     - Flag indicating whether to zero pad, or pad with existing data
   * - ``calc_mode_covariance``
     - -
     - True
     - Flag indicating whether to calculate the summary stats for the covariance matrix of the modes (later used for orthogonalizing scores of modes)
   * - ``use_HM``
     - -
     - True
     - Not yet implemented

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - If do_relative_binning 1. Array with summary stats for constant phase 2. Array with summary stats for linear phase 3. Array with bin edges adjusted to lie on a FFT grid 4. Absolute GPS time of right edge of waveform + 1 index = score 5. Array with hole corrections on self.time_sub 6. Array with valid inds on self.time_sub Else: 1. Array with fft of overlaps with fiducial one set to zero-lag 2. Frequencies of fft grid 3. Absolute GPS time of right edge of waveform + 1 index = score 4. Array with hole corrections on self.time_sub 5. Array with valid inds on self.time_sub

Docstring
---------

.. code-block:: text

   Defines summary statistics for convolution with relative waveforms
   The statistics correspond to convolution with the calpha waveform,
   without any subgrid shifts
   :param trigger: Trigger in the form of a row of a processed clist
   :param location:
       Tuple of length 2 with (linear-free time, calphas), used if trigger
       is not given
   :param relative_freq_bins:
       Array with bin edges for frequency interpolation
   :param dcalphas: Array with calpha spacings of finer grid to optimize
   :param dt: Shift in time (s) to allow
   :param delta: Phase allowed to accumulate within each bin
   :param subset_defined:
       Flag indicating if we already defined the required subset of data,
       used to save on FFTs
   :param relative_binning:
       Flag indicating whether we are using relative binning
   :param zero_pad:
       Flag indicating whether to zero pad, or pad with existing data
   :param calc_mode_covariance:
       Flag indicating whether to calculate the summary stats for the
       covariance matrix of the modes
       (later used for orthogonalizing scores of modes)
   :param use_HM: Not yet implemented
   :return:
       If do_relative_binning
           1. Array with summary stats for constant phase
           2. Array with summary stats for linear phase
           3. Array with bin edges adjusted to lie on a FFT grid
           4. Absolute GPS time of right edge of waveform + 1 index = score
           5. Array with hole corrections on self.time_sub
           6. Array with valid inds on self.time_sub
       Else:
           1. Array with fft of overlaps with fiducial one set to zero-lag
           2. Frequencies of fft grid
           3. Absolute GPS time of right edge of waveform + 1 index = score
           4. Array with hole corrections on self.time_sub
           5. Array with valid inds on self.time_sub
