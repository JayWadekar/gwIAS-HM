utils
=====

.. automodule:: utils

   
   .. rubric:: Functions

   .. autosummary::
   
      A_lm_halfinclin
      A_lm_inclin
      Y_lm
      Y_lm_halfinclin
      amend_indices
      asdf_fromfile
      band_filter
      before_write_save_old
      bincent
      bool2int
      change_filter_times_fd
      change_filter_times_td
      change_wf_fd_grid
      checkempty
      close_hdf5
      coherent_score
      colorbar
      condition_filter
      condition_filter_td
      create_chirp_mass_directory_dict
      define_coarser_mask
      delete_hdf5_datasets
      env_init_lines
      extract_filename
      extract_parts
      find_closest_coarse_calphas
      gen_step_fd
      get_HL_filenames
      get_O3_lsc_pe_samples
      get_coincident_json_filelist
      get_detector_fnames
      get_dirs
      get_dtype
      get_evname_from_tgps
      get_gwf_channel_names
      get_hdf5_file
      get_injection_details
      get_json_fname
      get_left_right_fnames
      get_lsc_event_times
      get_root_dirs
      get_run
      get_spectrum
      get_strain_fnames
      get_tgps_from_evname
      handle_missing_run
      hilbert_transform
      hole_edges
      hole_edges_to_mask
      incoherent_score
      index_after_removal
      index_limits
      interpolate_asd
      interpolate_wf_fd
      is_LIGO_valid
      is_close_to
      is_in_run
      is_within
      load_dict_from_hdf5_attrs
      load_pesummary_io
      m1_m2_mt
      m1_m2_s1_s2_to_chieff_chia
      make_LIGO_mask
      make_template_ids
      make_trigger_ids
      mass_conversion
      match
      mmap_h5
      multiprocessing
      notch_filter
      notch_filter_sos
      orthogonalize_split
      pars_from_pars
      plot_veto_details
      populate_magic_methods
      preprocess_wildcards
      q_and_eta
      read_hdf5_node
      remove_bad_times
      remove_old_versions
      rm_suffix
      s1z_s2z_chieff_chia_from_pars
      safe_concatenate
      safelen
      save_dict_to_hdf5_attrs
      scalar
      sigma_from_median
      sinc_interp_by_factor_of_2
      sinc_interp_x2D
      sine_gaussian
      splitarray
      standardize_run_name
      submask
      threshold_rv
      track_job
      unbias_split
      write_hdf5_node
   
   .. rubric:: Classes

   .. autosummary::
   
      CustomDefaultdict
      CustomHDF5File
      CustomHDF5Group
      EditableHDF5Dataset
      HDF5DatasetSubset
      NumpyEncoder
      TupleEncoder
   