triggers_single_detector_HM.TriggerList.gen_psd_drift_correction
================================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Computes the proper normalization for varying PSD, optionally returns correction, scores, and score-indices used at a particular index # Warning: Currently only the 22 wf is used # Warning: Doesn't apply a linear-free correction to the indices, ensure its use is consistent

Signature
---------

.. code-block:: python

   def gen_psd_drift_correction(self, wf_whitened_fd = None, calphas = None, jump = None, tol = params.PSD_DRIFT_TOL, avg = 'median', override_window_size = None, verbose = True, indices = None, return_only_at_indices = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wf_whitened_fd``
     - -
     - None
     - Array of size nmode x len(rfftfreq(fftsize)) with frequency domain waveform
   * - ``calphas``
     - -
     - None
     - Set of calphas for waveform, used if waveform itself isn't provided
   * - ``jump``
     - -
     - None
     - In unit of index (a reasonable choice is 1/dt) Ensure that jump is smaller than support_whitened_wf. If None, uses default in class (used this order to ensure that Matias's code works)
   * - ``tol``
     - -
     - params.PSD_DRIFT_TOL
     - Tolerance on correction factor^2 (stdev of this qty/mean)
   * - ``avg``
     - -
     - 'median'
     - Flag indicating the average to use. Can be one of mean, median, trimmedmean, and safemean
   * - ``override_window_size``
     - -
     - None
     - Override the window size (in indices)
   * - ``verbose``
     - -
     - True
     - Flag indicating whether to print details of computation
   * - ``indices``
     - -
     - None
     - Return window used to compute correction at indices within this list
   * - ``return_only_at_indices``
     - -
     - False
     - Flag to restrict computation to indices

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - if indices is not None if return_only_at_indices: PSD drift corrections at indices, array with [leftind, rightind] used for score at indices else: Array of length self.time with PSD drift correction array with [leftind, rightind] used for score at indices else: Array of length self.time with PSD drift correction

Docstring
---------

.. code-block:: text

   Computes the proper normalization for varying PSD, optionally
   returns correction, scores, and score-indices used at a particular index
   # Warning: Currently only the 22 wf is used
   # Warning: Doesn't apply a linear-free correction to the indices,
   ensure its use is consistent
   :param wf_whitened_fd:
       Array of size nmode x len(rfftfreq(fftsize)) with
        frequency domain waveform
   :param calphas:
       Set of calphas for waveform, used if waveform itself isn't provided
   :param jump:
       In unit of index (a reasonable choice is 1/dt) Ensure that jump is
       smaller than support_whitened_wf. If None, uses default in class
       (used this order to ensure that Matias's code works)
   :param tol: Tolerance on correction factor^2 (stdev of this qty/mean)
   :param avg:
       Flag indicating the average to use. Can be one of
       mean, median, trimmedmean, and safemean
   :param override_window_size: Override the window size (in indices)
   :param verbose: Flag indicating whether to print details of computation
   :param indices:
       Return window used to compute correction at indices within this list
   :param return_only_at_indices: Flag to restrict computation to indices
   :return: if indices is not None
               if return_only_at_indices:
                   PSD drift corrections at indices,
                   array with [leftind, rightind] used for score at indices
               else:
                   Array of length self.time with PSD drift correction
                   array with [leftind, rightind] used for score at indices
            else:
               Array of length self.time with PSD drift correction
