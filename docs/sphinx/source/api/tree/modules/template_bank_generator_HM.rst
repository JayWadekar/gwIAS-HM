template_bank_generator_HM
==========================

Back to :doc:`API tree index <../index>`

Purpose
-------

HM template-bank generation and basis-coefficient utilities.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`compute_hplus_hcross <../functions/template_bank_generator_HM.compute_hplus_hcross>`
     - Generate frequency domain waveform using LAL. Return hplus, hcross evaluated at f.
   * - :doc:`compute_overlap <../functions/template_bank_generator_HM.compute_overlap>`
     - Overlap between two waveforms
   * - :doc:`compute_snr_efficiency <../functions/template_bank_generator_HM.compute_snr_efficiency>`
     - Computes factor by which overlap degrades = SNR efficiency
   * - :doc:`gen_random_pars <../functions/template_bank_generator_HM.gen_random_pars>`
     - Generates n_wf random parameters. Applies either mrng (preferably) or mtrng, along with qrng
   * - :doc:`gen_waveform <../functions/template_bank_generator_HM.gen_waveform>`
     - Generates waveforms with given binary parameters
   * - :doc:`get_df <../functions/template_bank_generator_HM.get_df>`
     - No docstring summary available.
   * - :doc:`get_efficient_frequencies <../functions/template_bank_generator_HM.get_efficient_frequencies>`
     - Return a grid of frequencies that allows to safely get unwrapped phase. At low frequencies (fmin < f < fmid) it guarantees that the post-Newtonian phase increases by constant amounts of <= delta_radians for mchirp >= mchirp_min. At high frequencies (fmid < f < fmax) it switches to a regular grid.
   * - :doc:`get_prior_interp_func <../functions/template_bank_generator_HM.get_prior_interp_func>`
     - create a function that interpolate the prior for N-dimensional c-alpha array
   * - :doc:`remove_linear_component <../functions/template_bank_generator_HM.remove_linear_component>`
     - Fits out a linear dependence of phase w.r.t frequency
   * - :doc:`summarize_prior <../functions/template_bank_generator_HM.summarize_prior>`
     - This module creates the .pickle file for the template prior which is later used in ranking the candidates.
   * - :doc:`summarize_prior_single_subbank <../functions/template_bank_generator_HM.summarize_prior_single_subbank>`
     - calpha_samples and calpha_test_set should have sample weights as their last column (the sample weights are corresponding to volume sensitivity in the case of the HM search)
   * - :doc:`transform_basis <../functions/template_bank_generator_HM.transform_basis>`
     - Returns components of waveforms in the given basis
   * - :doc:`transform_pars <../functions/template_bank_generator_HM.transform_pars>`
     - Transforms [m1, m2, s1z, s2z, lambda1, lambda2] to commonly used parameters [mchirp, eta, chieff, chia, tilde{lambda}, delta tilde{lambda}]
   * - :doc:`upsample_lwfs <../functions/template_bank_generator_HM.upsample_lwfs>`
     - Upsample given lwfs to an output frequency grid Note: Amplitudes are linearly interpolated, with zeros outside the range Pad phases with edge values to avoid discontinuities at the edges

.. list-table:: Classes
   :header-rows: 1

   * - Class
     - Summary
   * - :doc:`MultiBank <../classes/template_bank_generator_HM.MultiBank>`
     - No class docstring summary available.
   * - :doc:`TemplateBank <../classes/template_bank_generator_HM.TemplateBank>`
     - No class docstring summary available.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../classes/template_bank_generator_HM.MultiBank
   ../classes/template_bank_generator_HM.TemplateBank
   ../functions/template_bank_generator_HM.compute_hplus_hcross
   ../functions/template_bank_generator_HM.compute_overlap
   ../functions/template_bank_generator_HM.compute_snr_efficiency
   ../functions/template_bank_generator_HM.gen_random_pars
   ../functions/template_bank_generator_HM.gen_waveform
   ../functions/template_bank_generator_HM.get_df
   ../functions/template_bank_generator_HM.get_efficient_frequencies
   ../functions/template_bank_generator_HM.get_prior_interp_func
   ../functions/template_bank_generator_HM.remove_linear_component
   ../functions/template_bank_generator_HM.summarize_prior
   ../functions/template_bank_generator_HM.summarize_prior_single_subbank
   ../functions/template_bank_generator_HM.transform_basis
   ../functions/template_bank_generator_HM.transform_pars
   ../functions/template_bank_generator_HM.upsample_lwfs
