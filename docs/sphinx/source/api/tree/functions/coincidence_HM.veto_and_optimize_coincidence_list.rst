coincidence_HM.veto_and_optimize_coincidence_list
=================================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Returns list of vetoed and optimized coincident candidates (w/ timeslides) and any extra information

Signature
---------

.. code-block:: python

   def veto_and_optimize_coincidence_list(bg_events, trig1, trig2, time_shift_tol, threshold_chi2, minimal_time_slide_jump, veto_triggers = True, min_veto_chi2 = 32, apply_threshold = True, origin = 0, n_cores = 1, opt_format = 'new', output_timeseries = True, output_coherent_score = True, score_reduction_timeseries = 10, score_reduction_max = 5, detectors = ('H1', 'L1'), score_func = utils.incoherent_score, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``bg_events``
     - -
     - -
     - n_candidate x 2 x len(Processedclist[0]) array with candidates
   * - ``trig1``
     - -
     - -
     - Trigger object 1
   * - ``trig2``
     - -
     - -
     - Trigger object 2
   * - ``time_shift_tol``
     - -
     - -
     - Width (s) of buckets to collect triggers into, the \\\`friends" of a trigger with the same calpha live within the same bucket
   * - ``threshold_chi2``
     - -
     - -
     - Threshold in sum(SNR^2) above which we consider triggers for the background list (or signal)
   * - ``minimal_time_slide_jump``
     - -
     - -
     - Jumps in timeslides
   * - ``veto_triggers``
     - -
     - True
     - Flag to veto triggers
   * - ``min_veto_chi2``
     - -
     - 32
     - Apply vetos to candidates above this SNR^2 in a single detector
   * - ``apply_threshold``
     - -
     - True
     - Flag to apply threshold on single-detector chi2 after optimizing
   * - ``origin``
     - -
     - 0
     - Origin to split the trigger times relative to
   * - ``n_cores``
     - -
     - 1
     - Number of cores to use for splitting the veto computations
   * - ``opt_format``
     - -
     - 'new'
     - How we choose the finer calpha grid, changed between O1 and O2 analyses Exposed here to replicate old runs if needed
   * - ``output_timeseries``
     - -
     - True
     - Flag to output timeseries for the candidates
   * - ``output_coherent_score``
     - -
     - True
     - Flag to compute the coherent score integral for the candidates
   * - ``score_reduction_timeseries``
     - -
     - 10
     - Restrict triggers in timeseries to the ones with single_detector_SNR^2 > (base trigger SNR^2) - this parameter
   * - ``score_reduction_max``
     - -
     - 5
     - Absolute reduction in SNR^2 from the peak value to be allowed for secondary peak in the function secondary_peak_reject()
   * - ``detectors``
     - -
     - ('H1', 'L1')
     - Tuple with names of the two detectors we will be running coincidence with
   * - ``score_func``
     - -
     - utils.incoherent_score
     - Function to use to decide on the representative trigger (once we use the coherent score integral, this choice becomes unimportant)
   * - ``\*\*kwargs``
     - -
     - -
     - Extra arguments to score_func, if needed

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. n_candidates x 2 x len(processedclist[0]) array with optimized coincident triggers 2. Mask into coincident triggers that identifies triggers that survived 3. Boolean array of shape n_candidate x 2 x (len(self.outlier_reasons) + 11 + 2 \\\* len(split_chunks)) with metadata. Indices represent 0: CBC_CAT2 flag ("and" of the values for the cloud) 1: CBC_CAT3 flag ("and" of the values for the cloud) The 2:len(self.outlier_reasons) + 10 + 2\\\*len(split_chunks) elements have zeros marking glitch tests that fired 2: len(self.outlier_reasons) + 2: index into outlier reasons for excess-power-like tests len(self.outlier_reasons) + 2: Finer PSD drift killed it len(self.outlier_reasons) + 3: No chunks present for phase tests len(self.outlier_reasons) + 4: Overall chi-2 test len(self.outlier_reasons) + 5: len(self.outlier_reasons) + 5 + len(split_chunks): Split tests len(outlier_reasons) + 5 + len(split_chunks): Finer sinc-interpolation len(outlier_reasons) + 6 + len(split_chunks): No chunks present for stringent phase test len(outlier_reasons) + 7 + len(split_chunks): Stringent chi-2 test len(outlier_reasons) + 8 + len(split_chunks): len(outlier_reasons) + 8 + 2\\\*len(split_chunks): Stringent split tests len(outlier_reasons) + 8 + 2\\\*len(split_chunks): Not enough chunks present for chi2 test with higher nchunk len(outlier_reasons) + 9 + 2\\\*len(split_chunks): chi2 test with higher nchunk len(outlier_reasons) + 10 + 2\\\*len(split_chunks): Found another louder trigger in the same time-shift-tol window 4. If output_timeseries, list of 2-tuples with H1 and L1 timeseries for each candidate 5. If output_coherent_score, array with coherent scores for each candidiate 6. Text keys for Boolean array for quickly reading off which test failed

Docstring
---------

.. code-block:: text

   Returns list of vetoed and optimized coincident candidates (w/ timeslides)
   and any extra information
   :param bg_events:
       n_candidate x 2 x len(Processedclist[0]) array with candidates
   :param trig1: Trigger object 1
   :param trig2: Trigger object 2
   :param time_shift_tol:
       Width (s) of buckets to collect triggers into, the `friends" of a
       trigger with the same calpha live within the same bucket
   :param threshold_chi2:
       Threshold in sum(SNR^2) above which we consider triggers for the
       background list (or signal)
   :param minimal_time_slide_jump: Jumps in timeslides
   :param veto_triggers: Flag to veto triggers
   :param min_veto_chi2:
       Apply vetos to candidates above this SNR^2 in a single detector
   :param apply_threshold:
       Flag to apply threshold on single-detector chi2 after optimizing
   :param origin: Origin to split the trigger times relative to
   :param n_cores: Number of cores to use for splitting the veto computations
   :param opt_format:
       How we choose the finer calpha grid, changed between O1 and O2 analyses
       Exposed here to replicate old runs if needed
   :param output_timeseries: Flag to output timeseries for the candidates
   :param output_coherent_score:
       Flag to compute the coherent score integral for the candidates
   :param score_reduction_timeseries:
       Restrict triggers in timeseries to the ones with
       single_detector_SNR^2 > (base trigger SNR^2) - this parameter
   :param score_reduction_max:
       Absolute reduction in SNR^2 from the peak value to be allowed
       for secondary peak in the function secondary_peak_reject()
   :param detectors:
       Tuple with names of the two detectors we will be running coincidence
       with
   :param score_func:
       Function to use to decide on the representative trigger
       (once we use the coherent score integral, this choice becomes
       unimportant)
   :param kwargs: Extra arguments to score_func, if needed
   :return:
       1. n_candidates x 2 x len(processedclist[0]) array with optimized
          coincident triggers
       2. Mask into coincident triggers that identifies triggers that survived
       3. Boolean array of shape n_candidate x 2 x
           (len(self.outlier_reasons) + 11 + 2 * len(split_chunks))
           with metadata. Indices represent
           0: CBC_CAT2 flag ("and" of the values for the cloud)
           1: CBC_CAT3 flag ("and" of the values for the cloud)
           The 2:len(self.outlier_reasons) + 10 + 2*len(split_chunks) elements
           have zeros marking glitch tests that fired
           2: len(self.outlier_reasons) + 2: index into outlier reasons
               for excess-power-like tests
           len(self.outlier_reasons) + 2: Finer PSD drift killed it
           len(self.outlier_reasons) + 3: No chunks present for phase tests
           len(self.outlier_reasons) + 4: Overall chi-2 test
           len(self.outlier_reasons) + 5:
               len(self.outlier_reasons) + 5 + len(split_chunks): Split tests
           len(outlier_reasons) + 5 + len(split_chunks):
               Finer sinc-interpolation
           len(outlier_reasons) + 6 + len(split_chunks):
               No chunks present for stringent phase test
           len(outlier_reasons) + 7 + len(split_chunks): Stringent chi-2 test
           len(outlier_reasons) + 8 + len(split_chunks):
               len(outlier_reasons) + 8 + 2*len(split_chunks):
                   Stringent split tests
           len(outlier_reasons) + 8 + 2*len(split_chunks):
               Not enough chunks present for chi2 test with higher nchunk
           len(outlier_reasons) + 9 + 2*len(split_chunks):
               chi2 test with higher nchunk
           len(outlier_reasons) + 10 + 2*len(split_chunks):
               Found another louder trigger in the same time-shift-tol window
       4. If output_timeseries, list of 2-tuples with H1 and L1 timeseries for 
          each candidate
       5. If output_coherent_score, array with coherent scores for each
           candidiate
       6. Text keys for Boolean array for quickly reading off which test failed
