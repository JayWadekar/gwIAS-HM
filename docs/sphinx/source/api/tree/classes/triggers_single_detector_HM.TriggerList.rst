triggers_single_detector_HM.TriggerList
=======================================

Back to :doc:`Module page <../modules/triggers_single_detector_HM>`

Summary
-------

Class that takes in directories with data and template bank information, and generates/analyzes triggers # TODO: Remove notch_wt_filter options etc

.. list-table:: Methods
   :header-rows: 1

   * - Method
     - Summary
   * - :doc:`empty_init <../methods/triggers_single_detector_HM.TriggerList.empty_init>`
     - No docstring summary available.
   * - :doc:`from_json <../methods/triggers_single_detector_HM.TriggerList.from_json>`
     - Instantiates class from json file created by previous run/gen_triggers Load using load_data=True and do_signal_processing=False to access the data without any glitch rejection
   * - :doc:`from_gwosc <../methods/triggers_single_detector_HM.TriggerList.from_gwosc>`
     - Loads data from gwosc and makes a trigger file, records the event time as given by LIGO (different from the linear-free time) in t_event Make sure to pass an appropriate template_conf for the parameters of the event in load_kwargs If we are doing something custom (e.g., adding holes etc), save using to_json along with fname_preprocessing, and we can reload it later with the desired modifications passed to from_json
   * - :doc:`load_data_from_preprocessing <../methods/triggers_single_detector_HM.TriggerList.load_data_from_preprocessing>`
     - Load preprocessed data from file
   * - :doc:`to_json <../methods/triggers_single_detector_HM.TriggerList.to_json>`
     - Saves information about trigger list to json file
   * - :doc:`save_preprocessing <../methods/triggers_single_detector_HM.TriggerList.save_preprocessing>`
     - Save preprocessing to file
   * - :doc:`save_candidatelist <../methods/triggers_single_detector_HM.TriggerList.save_candidatelist>`
     - -
   * - :doc:`safe_set_waveform_conditioning <../methods/triggers_single_detector_HM.TriggerList.safe_set_waveform_conditioning>`
     - Sets waveform conditioning given the ASD, and expands fftsize by a factor of two if needed to make it work. Useful when trialing small FFTsizes for BNS-like banks
   * - :doc:`pack_trigs <../methods/triggers_single_detector_HM.TriggerList.pack_trigs>`
     - No docstring summary available.
   * - :doc:`unpack_trigs <../methods/triggers_single_detector_HM.TriggerList.unpack_trigs>`
     - No docstring summary available.
   * - :doc:`inject_wf_into_data <../methods/triggers_single_detector_HM.TriggerList.inject_wf_into_data>`
     - Injects waveform into strain data, read description of injection_args for time convention, modifies strain in place as strain is a mutable type
   * - :doc:`get_glitch_thresholds <../methods/triggers_single_detector_HM.TriggerList.get_glitch_thresholds>`
     - Function giving test-statistic values for various glitch tests attained in the presence of noiseless waveforms
   * - :doc:`hole_snr_correction <../methods/triggers_single_detector_HM.TriggerList.hole_snr_correction>`
     - Computes SNR loss due to the mask (with appropriately inpainted data) Uses the stationary phase approximation, so valid only when the frequencies >> 1/hole size. We do not assume waveforms are normalized. Note: extended by Jay to include corrections due to bands in mask_stft
   * - :doc:`gen_psd_drift_correction <../methods/triggers_single_detector_HM.TriggerList.gen_psd_drift_correction>`
     - Computes the proper normalization for varying PSD, optionally returns correction, scores, and score-indices used at a particular index # Warning: Currently only the 22 wf is used # Warning: Doesn't apply a linear-free correction to the indices, ensure its use is consistent
   * - :doc:`prepare_for_triggers <../methods/triggers_single_detector_HM.TriggerList.prepare_for_triggers>`
     - Set things up to generate triggers
   * - :doc:`prepare_subset_for_triggers <../methods/triggers_single_detector_HM.TriggerList.prepare_subset_for_triggers>`
     - Prepare to compute everything on a reduced set of data to save time for BH-like waveforms. Defines subset of data, performs relevant FFTs, and stores in class variables Note that if relevant_index + right_inds_scores goes past the edge of the data, then we will have zeros in all the elements for which we don't have data
   * - :doc:`scores_wf <../methods/triggers_single_detector_HM.TriggerList.scores_wf>`
     - Computes overlaps and hole corrections for multiple waveforms We do not assume waveforms are normalized
   * - :doc:`gen_scores <../methods/triggers_single_detector_HM.TriggerList.gen_scores>`
     - Convenience function to generate overlaps and hole corrections, for multiple waveforms or calphas, on full data or a subset of it
   * - :doc:`gen_overlaps <../methods/triggers_single_detector_HM.TriggerList.gen_overlaps>`
     - Convenience function to generate hole-corrected overlaps on the whole file The first three parameters are prefered in order of appearance
   * - :doc:`process_clist <../methods/triggers_single_detector_HM.TriggerList.process_clist>`
     - Applies correction factor to times and SNRs and makes real array clist is assumed to have the following columns: if self.save_hole_correction is True: time, (overlap.real, overlap.imag) x 3, hole_correction x 3, psd_drift_correction, basis_coeff1, basis_coeff2, ... else: time, (overlap.real, overlap.imag) x 3, psd_drift_correction, basis_coeff1, basis_coeff2, ...
   * - :doc:`filter_with_wf <../methods/triggers_single_detector_HM.TriggerList.filter_with_wf>`
     - Generate triggers with sinc-interpolated scores with a single template for entire data or a subset of it
   * - :doc:`gen_triggers <../methods/triggers_single_detector_HM.TriggerList.gen_triggers>`
     - Defines a template bank with given spacing, performs matched filtering of data against this bank, and saves triggers above threshold
   * - :doc:`get_bad_times <../methods/triggers_single_detector_HM.TriggerList.get_bad_times>`
     - Finds bad times for the file, where it overproduces triggers due to widespread problems TODO: Fix this using preserve_max_snr?
   * - :doc:`get_time_index <../methods/triggers_single_detector_HM.TriggerList.get_time_index>`
     - "
   * - :doc:`prepare_subset_for_vetoes <../methods/triggers_single_detector_HM.TriggerList.prepare_subset_for_vetoes>`
     - Prepare subset that is large enough for vetos/optimization of any trigger in triggers
   * - :doc:`get_bestfit_wf <../methods/triggers_single_detector_HM.TriggerList.get_bestfit_wf>`
     - -
   * - :doc:`finer_psd_drift <../methods/triggers_single_detector_HM.TriggerList.finer_psd_drift>`
     - -
   * - :doc:`veto_trigger_power <../methods/triggers_single_detector_HM.TriggerList.veto_trigger_power>`
     - Checks trigger by subtracting best fit waveform and looking for excess power commensurate with waveform
   * - :doc:`veto_trigger_phase <../methods/triggers_single_detector_HM.TriggerList.veto_trigger_phase>`
     - Chi2 and tail veto using phase
   * - :doc:`veto_trigger_all <../methods/triggers_single_detector_HM.TriggerList.veto_trigger_all>`
     - Convenience function to apply all vetos
   * - :doc:`prepare_subset_for_optimization <../methods/triggers_single_detector_HM.TriggerList.prepare_subset_for_optimization>`
     - Defines subset if needed, and returns parameters to locate trigger in it
   * - :doc:`define_finer_grid_func <../methods/triggers_single_detector_HM.TriggerList.define_finer_grid_func>`
     - -
   * - :doc:`prepare_summary_stats <../methods/triggers_single_detector_HM.TriggerList.prepare_summary_stats>`
     - Defines summary statistics for convolution with relative waveforms The statistics correspond to convolution with the calpha waveform, without any subgrid shifts
   * - :doc:`gen_triggers_local <../methods/triggers_single_detector_HM.TriggerList.gen_triggers_local>`
     - Generates triggers at/on a small calpha grid around a trigger Has some not so quantifiable losses/biases due to the truncation of the waveforms that are being compared, and interpolations of the hole and PSD drift corrections Note: If the location is beyond the length of the data, it actually generates triggers around the edge due to quirks of searchsorted
   * - :doc:`gen_triggers_local_pars <../methods/triggers_single_detector_HM.TriggerList.gen_triggers_local_pars>`
     - NOTE: This fn has not been modified by Jay for higher modes. Generates triggers at/on a small calpha grid around a trigger Has some not so quantifiable losses/biases due to the truncation of the waveforms that are being compared, and interpolations of the hole and PSD drift corrections
   * - :doc:`filter_triggers <../methods/triggers_single_detector_HM.TriggerList.filter_triggers>`
     - Apply cuts on triggers in self.filteredclist. Call with filters=None, rejects=None, and reset_filters=True to clear filters
   * - :doc:`clip_outliers <../methods/triggers_single_detector_HM.TriggerList.clip_outliers>`
     - No docstring summary available.
   * - :doc:`clear_filter <../methods/triggers_single_detector_HM.TriggerList.clear_filter>`
     - No docstring summary available.
   * - :doc:`filter_processed_clist <../methods/triggers_single_detector_HM.TriggerList.filter_processed_clist>`
     - Filters a processed clist
   * - :doc:`remove_loud_times <../methods/triggers_single_detector_HM.TriggerList.remove_loud_times>`
     - No docstring summary available.
   * - :doc:`specgram <../methods/triggers_single_detector_HM.TriggerList.specgram>`
     - Makes spectrogram of whitened strain data
   * - :doc:`plot_teststat_hist <../methods/triggers_single_detector_HM.TriggerList.plot_teststat_hist>`
     - Plots normalized overall histogram of test statistic
   * - :doc:`plot_time_hist <../methods/triggers_single_detector_HM.TriggerList.plot_time_hist>`
     - Plots histogram of trigger time offsets from start of file (self.t0)
   * - :doc:`plot_triggers_corner <../methods/triggers_single_detector_HM.TriggerList.plot_triggers_corner>`
     - Make corner plot of parameters of triggers
   * - :doc:`plot_running_hist <../methods/triggers_single_detector_HM.TriggerList.plot_running_hist>`
     - TODO: Deprecation Warning Plots running histogram of test statistic in time-chunks
   * - :doc:`plot_bestfit_waveform <../methods/triggers_single_detector_HM.TriggerList.plot_bestfit_waveform>`
     - Plot waveforms in trigger(s) against the data

Class docstring
---------------

.. code-block:: text

   Class that takes in directories with data and template bank information,
   and generates/analyzes triggers
   # TODO: Remove notch_wt_filter options etc

.. toctree::
   :hidden:
   :maxdepth: 1

   ../methods/triggers_single_detector_HM.TriggerList.empty_init
   ../methods/triggers_single_detector_HM.TriggerList.from_json
   ../methods/triggers_single_detector_HM.TriggerList.from_gwosc
   ../methods/triggers_single_detector_HM.TriggerList.load_data_from_preprocessing
   ../methods/triggers_single_detector_HM.TriggerList.to_json
   ../methods/triggers_single_detector_HM.TriggerList.save_preprocessing
   ../methods/triggers_single_detector_HM.TriggerList.save_candidatelist
   ../methods/triggers_single_detector_HM.TriggerList.safe_set_waveform_conditioning
   ../methods/triggers_single_detector_HM.TriggerList.pack_trigs
   ../methods/triggers_single_detector_HM.TriggerList.unpack_trigs
   ../methods/triggers_single_detector_HM.TriggerList.inject_wf_into_data
   ../methods/triggers_single_detector_HM.TriggerList.get_glitch_thresholds
   ../methods/triggers_single_detector_HM.TriggerList.hole_snr_correction
   ../methods/triggers_single_detector_HM.TriggerList.gen_psd_drift_correction
   ../methods/triggers_single_detector_HM.TriggerList.prepare_for_triggers
   ../methods/triggers_single_detector_HM.TriggerList.prepare_subset_for_triggers
   ../methods/triggers_single_detector_HM.TriggerList.scores_wf
   ../methods/triggers_single_detector_HM.TriggerList.gen_scores
   ../methods/triggers_single_detector_HM.TriggerList.gen_overlaps
   ../methods/triggers_single_detector_HM.TriggerList.process_clist
   ../methods/triggers_single_detector_HM.TriggerList.filter_with_wf
   ../methods/triggers_single_detector_HM.TriggerList.gen_triggers
   ../methods/triggers_single_detector_HM.TriggerList.get_bad_times
   ../methods/triggers_single_detector_HM.TriggerList.get_time_index
   ../methods/triggers_single_detector_HM.TriggerList.prepare_subset_for_vetoes
   ../methods/triggers_single_detector_HM.TriggerList.get_bestfit_wf
   ../methods/triggers_single_detector_HM.TriggerList.finer_psd_drift
   ../methods/triggers_single_detector_HM.TriggerList.veto_trigger_power
   ../methods/triggers_single_detector_HM.TriggerList.veto_trigger_phase
   ../methods/triggers_single_detector_HM.TriggerList.veto_trigger_all
   ../methods/triggers_single_detector_HM.TriggerList.prepare_subset_for_optimization
   ../methods/triggers_single_detector_HM.TriggerList.define_finer_grid_func
   ../methods/triggers_single_detector_HM.TriggerList.prepare_summary_stats
   ../methods/triggers_single_detector_HM.TriggerList.gen_triggers_local
   ../methods/triggers_single_detector_HM.TriggerList.gen_triggers_local_pars
   ../methods/triggers_single_detector_HM.TriggerList.filter_triggers
   ../methods/triggers_single_detector_HM.TriggerList.clip_outliers
   ../methods/triggers_single_detector_HM.TriggerList.clear_filter
   ../methods/triggers_single_detector_HM.TriggerList.filter_processed_clist
   ../methods/triggers_single_detector_HM.TriggerList.remove_loud_times
   ../methods/triggers_single_detector_HM.TriggerList.specgram
   ../methods/triggers_single_detector_HM.TriggerList.plot_teststat_hist
   ../methods/triggers_single_detector_HM.TriggerList.plot_time_hist
   ../methods/triggers_single_detector_HM.TriggerList.plot_triggers_corner
   ../methods/triggers_single_detector_HM.TriggerList.plot_running_hist
   ../methods/triggers_single_detector_HM.TriggerList.plot_bestfit_waveform
