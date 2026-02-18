utils
=====

Back to :doc:`API tree index <../index>`

Purpose
-------

Cross-cutting utility functions and run/path helpers.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`A_lm_halfinclin <../functions/utils.A_lm_halfinclin>`
     - No docstring summary available.
   * - :doc:`A_lm_inclin <../functions/utils.A_lm_inclin>`
     - No docstring summary available.
   * - :doc:`amend_indices <../functions/utils.amend_indices>`
     - Joins blocks together if they are separated by less than n_tol
   * - :doc:`asdf_fromfile <../functions/utils.asdf_fromfile>`
     - -
   * - :doc:`band_filter <../functions/utils.band_filter>`
     - Creates desired filter, and computes its impulse response length
   * - :doc:`before_write_save_old <../functions/utils.before_write_save_old>`
     - No docstring summary available.
   * - :doc:`bincent <../functions/utils.bincent>`
     - No docstring summary available.
   * - :doc:`bool2int <../functions/utils.bool2int>`
     - No docstring summary available.
   * - :doc:`change_filter_times_fd <../functions/utils.change_filter_times_fd>`
     - Converts a conditioned filter to a different time array by zero-padding in time-domain
   * - :doc:`change_filter_times_td <../functions/utils.change_filter_times_td>`
     - Converts a conditioned filter to a different time array by zero-padding in time-domain
   * - :doc:`change_wf_fd_grid <../functions/utils.change_wf_fd_grid>`
     - -
   * - :doc:`checkempty <../functions/utils.checkempty>`
     - No docstring summary available.
   * - :doc:`close_hdf5 <../functions/utils.close_hdf5>`
     - -
   * - :doc:`coherent_score <../functions/utils.coherent_score>`
     - Combine H1 terms from scores_vetoed_max, returns scalar for single element
   * - :doc:`colorbar <../functions/utils.colorbar>`
     - No docstring summary available.
   * - :doc:`condition_filter <../functions/utils.condition_filter>`
     - Compute support, weight, and truncate input frequency domain filter
   * - :doc:`condition_filter_td <../functions/utils.condition_filter_td>`
     - Compute support, weight, and truncate input time domain filter
   * - :doc:`create_chirp_mass_directory_dict <../functions/utils.create_chirp_mass_directory_dict>`
     - Reads the names of all subdirectories, and creates a dictionary of dictionaries, with the outer dictionary having chirp mass ids as keys and the inner dictionary having subbank ids as keys and subdirectory names as values
   * - :doc:`define_coarser_mask <../functions/utils.define_coarser_mask>`
     - -
   * - :doc:`delete_hdf5_datasets <../functions/utils.delete_hdf5_datasets>`
     - Convenience function to prune some leaves from a hdf5 file
   * - :doc:`env_init_lines <../functions/utils.env_init_lines>`
     - Creates text snippet to be added before submitting jobs to clusters
   * - :doc:`extract_filename <../functions/utils.extract_filename>`
     - Extracts the filename from a string representation of a buffer object
   * - :doc:`extract_parts <../functions/utils.extract_parts>`
     - Extracts the parts of the directory name based on the prefix and suffix
   * - :doc:`find_closest_coarse_calphas <../functions/utils.find_closest_coarse_calphas>`
     - Finds the closest associated coarse calphas to given fine calphas
   * - :doc:`gen_step_fd <../functions/utils.gen_step_fd>`
     - No docstring summary available.
   * - :doc:`get_coincident_json_filelist <../functions/utils.get_coincident_json_filelist>`
     - Assumes that the file names go like "Det-Detn-...." where Detn is the key
   * - :doc:`get_detector_fnames <../functions/utils.get_detector_fnames>`
     - -
   * - :doc:`get_dirs <../functions/utils.get_dirs>`
     - Gives a list of length n_runs with each entry being a dictionary of dictionaries with the outer dictionary having chirp mass ids as keys and the inner dictionary having subbank ids as keys and subdirectory names as values. Figures out the number of subbanks and chirp mass ids by itself
   * - :doc:`get_dtype <../functions/utils.get_dtype>`
     - No docstring summary available.
   * - :doc:`get_evname_from_tgps <../functions/utils.get_evname_from_tgps>`
     - No docstring summary available.
   * - :doc:`get_gwf_channel_names <../functions/utils.get_gwf_channel_names>`
     - No docstring summary available.
   * - :doc:`get_hdf5_file <../functions/utils.get_hdf5_file>`
     - Returns a hdf5 file object with the given mode, given a source. Warning, if the file is already open in read-only mode, we can't return a writeable version.
   * - :doc:`get_HL_filenames <../functions/utils.get_HL_filenames>`
     - No docstring summary available.
   * - :doc:`get_injection_details <../functions/utils.get_injection_details>`
     - No docstring summary available.
   * - :doc:`get_json_fname <../functions/utils.get_json_fname>`
     - TODO: What if we choose a different fmax? Assumes that the file names go like "Det-Detn-...." where Detn is the detector key
   * - :doc:`get_left_right_fnames <../functions/utils.get_left_right_fnames>`
     - No docstring summary available.
   * - :doc:`get_lsc_event_times <../functions/utils.get_lsc_event_times>`
     - No docstring summary available.
   * - :doc:`get_O3_lsc_pe_samples <../functions/utils.get_O3_lsc_pe_samples>`
     - Returns a PESummary object to interact with the LSC PE samples
   * - :doc:`get_root_dirs <../functions/utils.get_root_dirs>`
     - Gets root directories by run, used since O3a is on scratch
   * - :doc:`get_run <../functions/utils.get_run>`
     - Returns the run name for a given GPS time
   * - :doc:`get_spectrum <../functions/utils.get_spectrum>`
     - Function used to dechirp Gets a trigger in the format of trigger, bank_id
   * - :doc:`get_strain_fnames <../functions/utils.get_strain_fnames>`
     - Returns the strain file names given a time
   * - :doc:`get_tgps_from_evname <../functions/utils.get_tgps_from_evname>`
     - No docstring summary available.
   * - :doc:`handle_missing_run <../functions/utils.handle_missing_run>`
     - No docstring summary available.
   * - :doc:`hilbert_transform <../functions/utils.hilbert_transform>`
     - No docstring summary available.
   * - :doc:`hole_edges <../functions/utils.hole_edges>`
     - No docstring summary available.
   * - :doc:`hole_edges_to_mask <../functions/utils.hole_edges_to_mask>`
     - No docstring summary available.
   * - :doc:`incoherent_score <../functions/utils.incoherent_score>`
     - -
   * - :doc:`index_after_removal <../functions/utils.index_after_removal>`
     - No docstring summary available.
   * - :doc:`index_limits <../functions/utils.index_limits>`
     - Function that decides which indices to include in the average such that we always average window_size indices to avoid \\\`regression-to-mean' artifacts due to fewer samples near holes
   * - :doc:`interpolate_asd <../functions/utils.interpolate_asd>`
     - get log-log interpolant of ASD with old_f[0] >= 0,
   * - :doc:`interpolate_wf_fd <../functions/utils.interpolate_wf_fd>`
     - get log-log interpolant of amp & lin of phase with old_f[0] >= 0,
   * - :doc:`is_close_to <../functions/utils.is_close_to>`
     - -
   * - :doc:`is_in_run <../functions/utils.is_in_run>`
     - Check if a GPS time is in a given run, works with HM/non-standard runs
   * - :doc:`is_LIGO_valid <../functions/utils.is_LIGO_valid>`
     - No docstring summary available.
   * - :doc:`is_LIGO_valid_between <../functions/utils.is_LIGO_valid_between>`
     - Take two gps times t1 and t2 and return: 0 if the hole chunk from t1 to t2 is invalid 1 if it is partially valid 2 if it is fully valid where valid means that both H and L have all the ['DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3'] flags True. (e.g. for assessing whether a waveform between t1 and t2 had the right to be found or not)
   * - :doc:`is_within <../functions/utils.is_within>`
     - No docstring summary available.
   * - :doc:`load_dict_from_hdf5_attrs <../functions/utils.load_dict_from_hdf5_attrs>`
     - Loads a dictionary from the attributes of a hdf5 file
   * - :doc:`load_pesummary_io <../functions/utils.load_pesummary_io>`
     - I need to do some monkey patching to make it work on my machine
   * - :doc:`m1_m2_mt <../functions/utils.m1_m2_mt>`
     - complete m1, m2, mt or check consistency if all given --> return m1, m2, mt
   * - :doc:`m1_m2_s1_s2_to_chieff_chia <../functions/utils.m1_m2_s1_s2_to_chieff_chia>`
     - No docstring summary available.
   * - :doc:`make_LIGO_mask <../functions/utils.make_LIGO_mask>`
     - Run this only once to make the file /data/bzackay/GW/LIGO_holes.npy
   * - :doc:`make_template_ids <../functions/utils.make_template_ids>`
     - No docstring summary available.
   * - :doc:`make_trigger_ids <../functions/utils.make_trigger_ids>`
     - No docstring summary available.
   * - :doc:`mass_conversion <../functions/utils.mass_conversion>`
     - Computes all conversions given a subset of parameters describing the masses
   * - :doc:`match <../functions/utils.match>`
     - Computes match, or cosine, between waveforms
   * - :doc:`mmap_h5 <../functions/utils.mmap_h5>`
     - Memory mapping is much faster for single row access than non-chunked hdf5, as long as we have a 64 bit architecture It is slower for random access, as the kernel version on the IAS systems is older and doesn't activate the madvise system call WARNING: If the hdf5 file is already open (e.g., in a writeable manner), the values in the mmap object might not be updated until the changes to the hdf5 file are flushed (despite the description of the dedault UNIX driver, it has a small buffer, set by rdcc_nbytes)!
   * - :doc:`multiprocessing <../functions/utils.multiprocessing>`
     - A shorthand function to apply multiprocessing in a one liner
   * - :doc:`notch_filter <../functions/utils.notch_filter>`
     - Creates notch filter
   * - :doc:`notch_filter_sos <../functions/utils.notch_filter_sos>`
     - Defines set of sos filters to apply to notch out lines
   * - :doc:`offset_background <../functions/utils.offset_background>`
     - Finds the amount to shift the detectors' data streams by. It returns an integer multiple of dt_shift
   * - :doc:`orthogonalize_split <../functions/utils.orthogonalize_split>`
     - Function to compute orthogonalized scores to those in the L subset
   * - :doc:`pars_from_pars <../functions/utils.pars_from_pars>`
     - WARNING: this sees keys 'chi1' and 'chi2' as being the vector spins [sjx, sjy, sjz] \\\*UNLIKE\\\* in other places where chij = sqrt(sjx^2 + sjy^2 + sjz^2) complete intrinsic parameter dictionary from sufficient parts
   * - :doc:`plot_veto_details <../functions/utils.plot_veto_details>`
     - Make four-panel diagnostic plot for vetoes
   * - :doc:`populate_magic_methods <../functions/utils.populate_magic_methods>`
     - Decorator to populate magic methods for HDF5DatasetSubset, they enable us to interact with it like it's a numpy array when needed
   * - :doc:`preprocess_wildcards <../functions/utils.preprocess_wildcards>`
     - No docstring summary available.
   * - :doc:`q_and_eta <../functions/utils.q_and_eta>`
     - No docstring summary available.
   * - :doc:`read_hdf5_node <../functions/utils.read_hdf5_node>`
     - Convenience function to read/create a group from a hdf5 file
   * - :doc:`remove_bad_times <../functions/utils.remove_bad_times>`
     - Removes elements of dat that are in the same bucket as in bad_time_list
   * - :doc:`remove_old_versions <../functions/utils.remove_old_versions>`
     - No docstring summary available.
   * - :doc:`rm_suffix <../functions/utils.rm_suffix>`
     - Utility to change the extension of a path
   * - :doc:`s1z_s2z_chieff_chia_from_pars <../functions/utils.s1z_s2z_chieff_chia_from_pars>`
     - \\\*NOTE\\\* this uses chia = (m1\\\*s1z - m2\\\*s2z) / (m1 + m2)
   * - :doc:`safe_concatenate <../functions/utils.safe_concatenate>`
     - No docstring summary available.
   * - :doc:`safelen <../functions/utils.safelen>`
     - No docstring summary available.
   * - :doc:`save_dict_to_hdf5_attrs <../functions/utils.save_dict_to_hdf5_attrs>`
     - Saves a dictionary to the attributes of a hdf5 file
   * - :doc:`scalar <../functions/utils.scalar>`
     - No docstring summary available.
   * - :doc:`sigma_from_median <../functions/utils.sigma_from_median>`
     - Computes sigma from median for an array with Gaussian samps + outliers Silently returns 1 if we passed in an empty array
   * - :doc:`sinc_interp_by_factor_of_2 <../functions/utils.sinc_interp_by_factor_of_2>`
     - -
   * - :doc:`sinc_interp_x2D <../functions/utils.sinc_interp_x2D>`
     - exact same as sinc_interp_by_factor_of_2() except along last axis of two dimensional array x2D
   * - :doc:`sine_gaussian <../functions/utils.sine_gaussian>`
     - Function to return time-domain sine-gaussian pulses (ready for FFT)
   * - :doc:`splitarray <../functions/utils.splitarray>`
     - Splits a single or multi-dimensional array into sub-arrays based on a coordinate
   * - :doc:`standardize_run_name <../functions/utils.standardize_run_name>`
     - Maintain consistency in run names, mainly used to confirm with the GWOSC naming conventions
   * - :doc:`submask <../functions/utils.submask>`
     - -
   * - :doc:`threshold_rv <../functions/utils.threshold_rv>`
     - -
   * - :doc:`track_job <../functions/utils.track_job>`
     - Track a multiprocessing job that was submitted in chunks
   * - :doc:`unbias_split <../functions/utils.unbias_split>`
     - Function to compute split scores - expectations from total score
   * - :doc:`write_hdf5_node <../functions/utils.write_hdf5_node>`
     - Convenience function to create/overwrite a leaf in a hdf5 file
   * - :doc:`Y_lm <../functions/utils.Y_lm>`
     - NOTE: azim = pi/2 - vphi
   * - :doc:`Y_lm_halfinclin <../functions/utils.Y_lm_halfinclin>`
     - NOTE: azim = pi/2 - vphi

.. list-table:: Classes
   :header-rows: 1

   * - Class
     - Summary
   * - :doc:`CustomDefaultdict <../classes/utils.CustomDefaultdict>`
     - No class docstring summary available.
   * - :doc:`CustomHDF5File <../classes/utils.CustomHDF5File>`
     - Custom h5py file object that wraps datasets with EditableHDF5Dataset on access
   * - :doc:`CustomHDF5Group <../classes/utils.CustomHDF5Group>`
     - Custom h5py group object that wraps datasets with EditableHDF5Dataset on access
   * - :doc:`EditableHDF5Dataset <../classes/utils.EditableHDF5Dataset>`
     - Wrapper of hdf5 dataset that returns references instead of copies when indexed if it is multidimensional or variable-length
   * - :doc:`HDF5DatasetSubset <../classes/utils.HDF5DatasetSubset>`
     - Simulates a copy-on-write reference to a subset of a multidimensional or variable-length HDF5 dataset. This is needed because by default, edits to a HDF5 dataset's elements don't propagate back to the underlying dataset
   * - :doc:`NumpyEncoder <../classes/utils.NumpyEncoder>`
     - No class docstring summary available.
   * - :doc:`TupleEncoder <../classes/utils.TupleEncoder>`
     - No class docstring summary available.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../classes/utils.CustomDefaultdict
   ../classes/utils.CustomHDF5File
   ../classes/utils.CustomHDF5Group
   ../classes/utils.EditableHDF5Dataset
   ../classes/utils.HDF5DatasetSubset
   ../classes/utils.NumpyEncoder
   ../classes/utils.TupleEncoder
   ../functions/utils.A_lm_halfinclin
   ../functions/utils.A_lm_inclin
   ../functions/utils.amend_indices
   ../functions/utils.asdf_fromfile
   ../functions/utils.band_filter
   ../functions/utils.before_write_save_old
   ../functions/utils.bincent
   ../functions/utils.bool2int
   ../functions/utils.change_filter_times_fd
   ../functions/utils.change_filter_times_td
   ../functions/utils.change_wf_fd_grid
   ../functions/utils.checkempty
   ../functions/utils.close_hdf5
   ../functions/utils.coherent_score
   ../functions/utils.colorbar
   ../functions/utils.condition_filter
   ../functions/utils.condition_filter_td
   ../functions/utils.create_chirp_mass_directory_dict
   ../functions/utils.define_coarser_mask
   ../functions/utils.delete_hdf5_datasets
   ../functions/utils.env_init_lines
   ../functions/utils.extract_filename
   ../functions/utils.extract_parts
   ../functions/utils.find_closest_coarse_calphas
   ../functions/utils.gen_step_fd
   ../functions/utils.get_coincident_json_filelist
   ../functions/utils.get_detector_fnames
   ../functions/utils.get_dirs
   ../functions/utils.get_dtype
   ../functions/utils.get_evname_from_tgps
   ../functions/utils.get_gwf_channel_names
   ../functions/utils.get_hdf5_file
   ../functions/utils.get_HL_filenames
   ../functions/utils.get_injection_details
   ../functions/utils.get_json_fname
   ../functions/utils.get_left_right_fnames
   ../functions/utils.get_lsc_event_times
   ../functions/utils.get_O3_lsc_pe_samples
   ../functions/utils.get_root_dirs
   ../functions/utils.get_run
   ../functions/utils.get_spectrum
   ../functions/utils.get_strain_fnames
   ../functions/utils.get_tgps_from_evname
   ../functions/utils.handle_missing_run
   ../functions/utils.hilbert_transform
   ../functions/utils.hole_edges
   ../functions/utils.hole_edges_to_mask
   ../functions/utils.incoherent_score
   ../functions/utils.index_after_removal
   ../functions/utils.index_limits
   ../functions/utils.interpolate_asd
   ../functions/utils.interpolate_wf_fd
   ../functions/utils.is_close_to
   ../functions/utils.is_in_run
   ../functions/utils.is_LIGO_valid
   ../functions/utils.is_LIGO_valid_between
   ../functions/utils.is_within
   ../functions/utils.load_dict_from_hdf5_attrs
   ../functions/utils.load_pesummary_io
   ../functions/utils.m1_m2_mt
   ../functions/utils.m1_m2_s1_s2_to_chieff_chia
   ../functions/utils.make_LIGO_mask
   ../functions/utils.make_template_ids
   ../functions/utils.make_trigger_ids
   ../functions/utils.mass_conversion
   ../functions/utils.match
   ../functions/utils.mmap_h5
   ../functions/utils.multiprocessing
   ../functions/utils.notch_filter
   ../functions/utils.notch_filter_sos
   ../functions/utils.offset_background
   ../functions/utils.orthogonalize_split
   ../functions/utils.pars_from_pars
   ../functions/utils.plot_veto_details
   ../functions/utils.populate_magic_methods
   ../functions/utils.preprocess_wildcards
   ../functions/utils.q_and_eta
   ../functions/utils.read_hdf5_node
   ../functions/utils.remove_bad_times
   ../functions/utils.remove_old_versions
   ../functions/utils.rm_suffix
   ../functions/utils.s1z_s2z_chieff_chia_from_pars
   ../functions/utils.safe_concatenate
   ../functions/utils.safelen
   ../functions/utils.save_dict_to_hdf5_attrs
   ../functions/utils.scalar
   ../functions/utils.sigma_from_median
   ../functions/utils.sinc_interp_by_factor_of_2
   ../functions/utils.sinc_interp_x2D
   ../functions/utils.sine_gaussian
   ../functions/utils.splitarray
   ../functions/utils.standardize_run_name
   ../functions/utils.submask
   ../functions/utils.threshold_rv
   ../functions/utils.track_job
   ../functions/utils.unbias_split
   ../functions/utils.write_hdf5_node
   ../functions/utils.Y_lm
   ../functions/utils.Y_lm_halfinclin
