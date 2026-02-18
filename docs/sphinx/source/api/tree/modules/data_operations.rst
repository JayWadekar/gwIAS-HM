data_operations
===============

Back to :doc:`API tree index <../index>`

Purpose
-------

PSD estimation, whitening, and glitch/line mitigation utilities.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`asd_func <../functions/data_operations.asd_func>`
     - Returns function that takes frequencies in Hz and gives the interpolated ASD in units of 1/Hz^0.5
   * - :doc:`band_eraser <../functions/data_operations.band_eraser>`
     - Detects bandlimited excess power transients
   * - :doc:`calculate_excess_power <../functions/data_operations.calculate_excess_power>`
     - Returns samples of excess power with and without a rolling average
   * - :doc:`calculate_sine_gaussian_overlaps <../functions/data_operations.calculate_sine_gaussian_overlaps>`
     - Returns sine gaussian scores
   * - :doc:`chunkedfft <../functions/data_operations.chunkedfft>`
     - Returns FFTs of chunks of data for overlap-save. Coordinates assumed dimensionless, as in DFT
   * - :doc:`data_to_asdfunc <../functions/data_operations.data_to_asdfunc>`
     - Passes the data to welch and returns a function that maps frequencies to ASD
   * - :doc:`define_crude_line_mask <../functions/data_operations.define_crude_line_mask>`
     - Marks lines in the PSD with zeros, errs on the side of marking lines
   * - :doc:`define_crude_line_mask_v2 <../functions/data_operations.define_crude_line_mask_v2>`
     - Function to flag lines in the PSD, errs on the side of marking lines
   * - :doc:`fill_hole_consecutive <../functions/data_operations.fill_hole_consecutive>`
     - -
   * - :doc:`fill_holes_bruteforce <../functions/data_operations.fill_holes_bruteforce>`
     - -
   * - :doc:`fill_holes_file <../functions/data_operations.fill_holes_file>`
     - -
   * - :doc:`fill_holes_segment <../functions/data_operations.fill_holes_segment>`
     - Fill holes in a segment of data
   * - :doc:`find_excess_power_transients <../functions/data_operations.find_excess_power_transients>`
     - Detects bandlimited excess power transients
   * - :doc:`find_sine_gaussian_transients <../functions/data_operations.find_sine_gaussian_transients>`
     - Finds sine-gaussian transients in data
   * - :doc:`find_whitened_outliers <../functions/data_operations.find_whitened_outliers>`
     - Zeros mask in place on either side of outliers in whitened data stream
   * - :doc:`gen_whitened_notched_strain <../functions/data_operations.gen_whitened_notched_strain>`
     - Whiten and notch, modify strain_wt_down in place, and perhaps identify varying lines in mask_freqs
   * - :doc:`loaddata <../functions/data_operations.loaddata>`
     - Load data Warning: 1. Assumes that the quality flags are sampled at 1 Hz 2. Assumes that the sampling rate of data is an integer # TODO: If mask is too intermittent, make it flat since it is unreliable...
   * - :doc:`makeasdplots <../functions/data_operations.makeasdplots>`
     - Makes comparison plots of ASDs with the data-derived one
   * - :doc:`median_bias <../functions/data_operations.median_bias>`
     - Returns the bias of the median of a set of periodograms relative to the mean. See arXiv:gr-qc/0509116 Appendix B for details.
   * - :doc:`norm_matched_filter_overlap <../functions/data_operations.norm_matched_filter_overlap>`
     - Computes matched filter overlap for a sliding template. Coordinates assumed dimensionless, as in DFT.
   * - :doc:`overlap_save <../functions/data_operations.overlap_save>`
     - Convolution with the overlap-save method. Look at the modes in chunkedfft to understand what transients are captured at the edges Coordinates assumed dimensionless, as in DFT Note: Overlap-save is only justified when the window is strictly compact and contained within wl
   * - :doc:`process_data <../functions/data_operations.process_data>`
     - Preferably send in 2\\\*\\\*N samples
   * - :doc:`robust_power_filter <../functions/data_operations.robust_power_filter>`
     - -
   * - :doc:`scipy_12_welch <../functions/data_operations.scipy_12_welch>`
     - Cheating to copy Scipy 1.2's Welch method, to access the average attribute Note: Assumes fs is an integer in converting times to mask indices
   * - :doc:`specgram_quality <../functions/data_operations.specgram_quality>`
     - Computes specgram of whitened data, along with list of bad time and frequency channels in the specgram (bad time channels are windows that overlap with zeros in mask, and bad frequency channels are varying lines or those beyond the analysis range)
   * - :doc:`update_masks <../functions/data_operations.update_masks>`
     - Convenience function to update a mask in a range, record where we nulled, and avoid some regions if needed

No public classes were found.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../functions/data_operations.asd_func
   ../functions/data_operations.band_eraser
   ../functions/data_operations.calculate_excess_power
   ../functions/data_operations.calculate_sine_gaussian_overlaps
   ../functions/data_operations.chunkedfft
   ../functions/data_operations.data_to_asdfunc
   ../functions/data_operations.define_crude_line_mask
   ../functions/data_operations.define_crude_line_mask_v2
   ../functions/data_operations.fill_hole_consecutive
   ../functions/data_operations.fill_holes_bruteforce
   ../functions/data_operations.fill_holes_file
   ../functions/data_operations.fill_holes_segment
   ../functions/data_operations.find_excess_power_transients
   ../functions/data_operations.find_sine_gaussian_transients
   ../functions/data_operations.find_whitened_outliers
   ../functions/data_operations.gen_whitened_notched_strain
   ../functions/data_operations.loaddata
   ../functions/data_operations.makeasdplots
   ../functions/data_operations.median_bias
   ../functions/data_operations.norm_matched_filter_overlap
   ../functions/data_operations.overlap_save
   ../functions/data_operations.process_data
   ../functions/data_operations.robust_power_filter
   ../functions/data_operations.scipy_12_welch
   ../functions/data_operations.specgram_quality
   ../functions/data_operations.update_masks
