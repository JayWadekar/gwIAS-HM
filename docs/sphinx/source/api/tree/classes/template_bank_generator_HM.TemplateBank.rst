template_bank_generator_HM.TemplateBank
=======================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

No class docstring summary available.

.. list-table:: Methods
   :header-rows: 1

   * - Method
     - Summary
   * - :doc:`from_json <../methods/template_bank_generator_HM.TemplateBank.from_json>`
     - Instantiates class from json file created by previous run
   * - :doc:`gen_amps_from_calpha <../methods/template_bank_generator_HM.TemplateBank.gen_amps_from_calpha>`
     - Generates amplitudes of all modes from given calphas
   * - :doc:`gen_phases_from_calpha <../methods/template_bank_generator_HM.TemplateBank.gen_phases_from_calpha>`
     - Generates phases of all modes from given coefficients
   * - :doc:`get_linear_free_shift_from_wf_fd <../methods/template_bank_generator_HM.TemplateBank.get_linear_free_shift_from_wf_fd>`
     - No docstring summary available.
   * - :doc:`get_linear_free_shift_from_pars <../methods/template_bank_generator_HM.TemplateBank.get_linear_free_shift_from_pars>`
     - No docstring summary available.
   * - :doc:`get_linear_free_shift_from_calpha <../methods/template_bank_generator_HM.TemplateBank.get_linear_free_shift_from_calpha>`
     - No docstring summary available.
   * - :doc:`gen_wf_td_from_pars <../methods/template_bank_generator_HM.TemplateBank.gen_wf_td_from_pars>`
     - WARNING: Not modified by Jay for the case of HM
   * - :doc:`gen_wf_fd_from_pars <../methods/template_bank_generator_HM.TemplateBank.gen_wf_fd_from_pars>`
     - Convenience interface to gen_waveform that removes the linear component of the phase in the same way as was used to make the bank
   * - :doc:`gen_wfs_td_from_fd <../methods/template_bank_generator_HM.TemplateBank.gen_wfs_td_from_fd>`
     - Generates conditioned time-domain waveforms given frequency domain waveforms.
   * - :doc:`gen_wfs_fd_from_calpha <../methods/template_bank_generator_HM.TemplateBank.gen_wfs_fd_from_calpha>`
     - Generates waveforms from given coefficients Note: At fs_out <= min(fs_basis) or >= max(fs_basis), the output amplitudes are zero
   * - :doc:`gen_wfs_td_from_calpha <../methods/template_bank_generator_HM.TemplateBank.gen_wfs_td_from_calpha>`
     - Generates time-domain waveforms given basis coefficients OK for FFT test but might not be for injection due to wraparound and truncation losses
   * - :doc:`orthogonalize_wfs <../methods/template_bank_generator_HM.TemplateBank.orthogonalize_wfs>`
     - Orthogonalizes the different higher modes in the wf (useful for later appropriately calculating inner product of data with diff modes)
   * - :doc:`get_waveform_conditioning <../methods/template_bank_generator_HM.TemplateBank.get_waveform_conditioning>`
     - Useful function to get conditioning parameters for whitened waveforms
   * - :doc:`set_waveform_conditioning <../methods/template_bank_generator_HM.TemplateBank.set_waveform_conditioning>`
     - Finds normalization factor, support, and shift for whitened waveforms in bank if they aren't already defined \\\*NOTE\\\* for PSD estimation, need: fftsize \\\* dt > chunktime
   * - :doc:`gen_whitened_wfs_td <../methods/template_bank_generator_HM.TemplateBank.gen_whitened_wfs_td>`
     - Generates whitened and conditioned time-domain waveforms given basis coefficients, or input waveforms
   * - :doc:`gen_boundary_whitened_wfs_td <../methods/template_bank_generator_HM.TemplateBank.gen_boundary_whitened_wfs_td>`
     - Generates boundary whitened and conditioned time-domain waveforms
   * - :doc:`split_whitened_wf_td <../methods/template_bank_generator_HM.TemplateBank.split_whitened_wf_td>`
     - Splits whitened time domain waveform into chunks with desired fractions of SNR^2, useful for vetoes
   * - :doc:`wt_waveform_generator <../methods/template_bank_generator_HM.TemplateBank.wt_waveform_generator>`
     - Generator that returns waveforms in bank and corresponding c_alphas
   * - :doc:`grid_range <../methods/template_bank_generator_HM.TemplateBank.grid_range>`
     - Returns a grid between min_val and max_val If force_zero=True: 0 is a gridpoint Spacing is <= d_val/2 at the edges Spacing is <= d_val in the bulk If force_zero=False: Spacing is <= d_val/2 at the edges Spacing is exactly d_val in the bulk
   * - :doc:`make_grid_axes <../methods/template_bank_generator_HM.TemplateBank.make_grid_axes>`
     - Return a list of arrays with the axes of a grid that covers all the physical waveforms used to generate the bank.
   * - :doc:`define_important_grid <../methods/template_bank_generator_HM.TemplateBank.define_important_grid>`
     - This was the code for removing calpha dimensions that were too small Now, we always allow = ndims
   * - :doc:`ntemplates <../methods/template_bank_generator_HM.TemplateBank.ntemplates>`
     - Convenience function to output number of templates in the bank
   * - :doc:`get_grid_index <../methods/template_bank_generator_HM.TemplateBank.get_grid_index>`
     - Get the multiindex of the gridpoint that most closely matches a point. Assumes that the grid is rectangular, but not necessarily regular.
   * - :doc:`remove_nonphysical_templates_JR <../methods/template_bank_generator_HM.TemplateBank.remove_nonphysical_templates_JR>`
     - This function was written by Javier but now superseded Normalizing flow. Return the set of points from a rectangular grid in component space that are close to a physical waveform. Assumes that irrelevant c_alphas are zero (in the extra dimensions).
   * - :doc:`remove_nonphysical_templates <../methods/template_bank_generator_HM.TemplateBank.remove_nonphysical_templates>`
     - Removes unphysical templates using the normalizing flow based on their astrophysical prior probability.
   * - :doc:`get_coeff_grid <../methods/template_bank_generator_HM.TemplateBank.get_coeff_grid>`
     - Return the set of points on which to place templates, an array of dimension ntemplates x ndims
   * - :doc:`gen_physical_grid <../methods/template_bank_generator_HM.TemplateBank.gen_physical_grid>`
     - Generate a list of physical templates and save it inside a dictionary as a class attribute. The key of the dictionary is a tuple (delta_calpha, fudge, force_zero)
   * - :doc:`is_physical <../methods/template_bank_generator_HM.TemplateBank.is_physical>`
     - Checks if the given calphas are physical, i.e. that they are in the trimmed grid that remove_nonphysical_templates outputs
   * - :doc:`optimize_calpha <../methods/template_bank_generator_HM.TemplateBank.optimize_calpha>`
     - Return the calpha on the fine axis that: - was under the responsibility of calpha_coarse - is closest to calpha
   * - :doc:`closest_coeff <../methods/template_bank_generator_HM.TemplateBank.closest_coeff>`
     - Return the values of the coefficients in coeff_grid that most closely match some input coefficients. Similar to rounding to the grid spacing, but accounts for offset and the grid needn't be uniform or rectangular, e.g. if nonphysical templates have been removed.
   * - :doc:`test_effectualness <../methods/template_bank_generator_HM.TemplateBank.test_effectualness>`
     - Computes the mismatches between original astrophysical waveforms and the bank's representation of them. Used to measure losses in sensitivity
   * - :doc:`def_relative_bins <../methods/template_bank_generator_HM.TemplateBank.def_relative_bins>`
     - Returns bin edges at which we evaluate relative waveforms
   * - :doc:`marginalized_HM_scores_incl_temp_prior <../methods/template_bank_generator_HM.TemplateBank.marginalized_HM_scores_incl_temp_prior>`
     - Adds the template prior to the marginalized HM scores Same arguments as marginalized_HM_scores() below
   * - :doc:`marginalized_HM_scores <../methods/template_bank_generator_HM.TemplateBank.marginalized_HM_scores>`
     - The scores obtained from triggering can sometimes be unphysical ,e.g., \\\|Z_33\\\|>>\\\|Z_22\\\| (here Z=complex rho timeseries) This function marginalizes or maximizes over inclination and mass ratio for HMs
   * - :doc:`get_approximate_params <../methods/template_bank_generator_HM.TemplateBank.get_approximate_params>`
     - Return a guess of m1, m2, s1z, s2z, l1, l2 from a template's coefficients We have currently only used 22, not HM (for which, breaking the q-Chieff degeneracy might help)
   * - :doc:`transform_pars <../methods/template_bank_generator_HM.TemplateBank.transform_pars>`
     - Transforms [m1, m2, s1z, s2z, lambda1, lambda2] to commonly used parameters [mchirp, eta, chieff, chia, tilde{lambda}, delta tilde{lambda}]
   * - :doc:`gen_random_wfs_td <../methods/template_bank_generator_HM.TemplateBank.gen_random_wfs_td>`
     - Has not been updated by Jay. Generates random time-domain waveforms, OK for FFT test but not for injection due to wraparound
   * - :doc:`gen_snr_degrade_dt <../methods/template_bank_generator_HM.TemplateBank.gen_snr_degrade_dt>`
     - Has not been updated by Jay. Computes SNR degradation for a number of random waveforms shifted by a single TD index, used in deciding the time resolution to use for the matched filtering
   * - :doc:`gen_phase_mismatch <../methods/template_bank_generator_HM.TemplateBank.gen_phase_mismatch>`
     - Has not been updated by Jay. Saves SNR degradation as a function of frequency for waveforms in bank. Note: Requires that bank was loaded with load_lwfs=True

Class docstring
---------------

No docstring text available.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../methods/template_bank_generator_HM.TemplateBank.from_json
   ../methods/template_bank_generator_HM.TemplateBank.gen_amps_from_calpha
   ../methods/template_bank_generator_HM.TemplateBank.gen_phases_from_calpha
   ../methods/template_bank_generator_HM.TemplateBank.get_linear_free_shift_from_wf_fd
   ../methods/template_bank_generator_HM.TemplateBank.get_linear_free_shift_from_pars
   ../methods/template_bank_generator_HM.TemplateBank.get_linear_free_shift_from_calpha
   ../methods/template_bank_generator_HM.TemplateBank.gen_wf_td_from_pars
   ../methods/template_bank_generator_HM.TemplateBank.gen_wf_fd_from_pars
   ../methods/template_bank_generator_HM.TemplateBank.gen_wfs_td_from_fd
   ../methods/template_bank_generator_HM.TemplateBank.gen_wfs_fd_from_calpha
   ../methods/template_bank_generator_HM.TemplateBank.gen_wfs_td_from_calpha
   ../methods/template_bank_generator_HM.TemplateBank.orthogonalize_wfs
   ../methods/template_bank_generator_HM.TemplateBank.get_waveform_conditioning
   ../methods/template_bank_generator_HM.TemplateBank.set_waveform_conditioning
   ../methods/template_bank_generator_HM.TemplateBank.gen_whitened_wfs_td
   ../methods/template_bank_generator_HM.TemplateBank.gen_boundary_whitened_wfs_td
   ../methods/template_bank_generator_HM.TemplateBank.split_whitened_wf_td
   ../methods/template_bank_generator_HM.TemplateBank.wt_waveform_generator
   ../methods/template_bank_generator_HM.TemplateBank.grid_range
   ../methods/template_bank_generator_HM.TemplateBank.make_grid_axes
   ../methods/template_bank_generator_HM.TemplateBank.define_important_grid
   ../methods/template_bank_generator_HM.TemplateBank.ntemplates
   ../methods/template_bank_generator_HM.TemplateBank.get_grid_index
   ../methods/template_bank_generator_HM.TemplateBank.remove_nonphysical_templates_JR
   ../methods/template_bank_generator_HM.TemplateBank.remove_nonphysical_templates
   ../methods/template_bank_generator_HM.TemplateBank.get_coeff_grid
   ../methods/template_bank_generator_HM.TemplateBank.gen_physical_grid
   ../methods/template_bank_generator_HM.TemplateBank.is_physical
   ../methods/template_bank_generator_HM.TemplateBank.optimize_calpha
   ../methods/template_bank_generator_HM.TemplateBank.closest_coeff
   ../methods/template_bank_generator_HM.TemplateBank.test_effectualness
   ../methods/template_bank_generator_HM.TemplateBank.def_relative_bins
   ../methods/template_bank_generator_HM.TemplateBank.marginalized_HM_scores_incl_temp_prior
   ../methods/template_bank_generator_HM.TemplateBank.marginalized_HM_scores
   ../methods/template_bank_generator_HM.TemplateBank.get_approximate_params
   ../methods/template_bank_generator_HM.TemplateBank.transform_pars
   ../methods/template_bank_generator_HM.TemplateBank.gen_random_wfs_td
   ../methods/template_bank_generator_HM.TemplateBank.gen_snr_degrade_dt
   ../methods/template_bank_generator_HM.TemplateBank.gen_phase_mismatch
