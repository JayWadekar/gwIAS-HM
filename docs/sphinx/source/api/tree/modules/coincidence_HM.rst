coincidence_HM
==============

Back to :doc:`API tree index <../index>`

Purpose
-------

Multi-detector coincidence, vetoing, and candidate output logic.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`collect_all_candidate_files_npy <../functions/coincidence_HM.collect_all_candidate_files_npy>`
     - -
   * - :doc:`collect_all_candidate_files_npz <../functions/coincidence_HM.collect_all_candidate_files_npz>`
     - -
   * - :doc:`collect_background_candidates <../functions/coincidence_HM.collect_background_candidates>`
     - Function to collect background candidates (including real events)
   * - :doc:`collect_files <../functions/coincidence_HM.collect_files>`
     - Returns list of length nfiles with loaded arrays, an element is None if the byte-size < 10
   * - :doc:`collect_files_npz <../functions/coincidence_HM.collect_files_npz>`
     - Returns list of length nfiles with each element being a tuple with events, mask_vetoed, veto_metadata, coherent_scores (+timeseries if collect_timeseries) an element is None if the byte-size < 10
   * - :doc:`combine_files <../functions/coincidence_HM.combine_files>`
     - Combines files that were read in using collect_files
   * - :doc:`combine_files_npz <../functions/coincidence_HM.combine_files_npz>`
     - Combines files that were read in using collect_files_npz
   * - :doc:`create_candidate_output_dir_name <../functions/coincidence_HM.create_candidate_output_dir_name>`
     - No docstring summary available.
   * - :doc:`create_shifted_observations <../functions/coincidence_HM.create_shifted_observations>`
     - No docstring summary available.
   * - :doc:`find_friends_in_cloud <../functions/coincidence_HM.find_friends_in_cloud>`
     - -
   * - :doc:`find_interesting_dir <../functions/coincidence_HM.find_interesting_dir>`
     - Goes through trigger files for H1 and L1 in a directory and performs coincidence analysis
   * - :doc:`find_interesting_dir_cluster <../functions/coincidence_HM.find_interesting_dir_cluster>`
     - -
   * - :doc:`fish_in_weaker_detector <../functions/coincidence_HM.fish_in_weaker_detector>`
     - Warning: This function has not been updated by Jay for HM case
   * - :doc:`fit_step_veto <../functions/coincidence_HM.fit_step_veto>`
     - -
   * - :doc:`flag_duplicates_per_group_pair <../functions/coincidence_HM.flag_duplicates_per_group_pair>`
     - Picks a subset of pair_trigger_array such that each bucket of group_size in H1 and L1 has a unique \\\`coincident' candidate. If a veto_mask is provided, defaults to preferentially keep in vetoed elements Note: The order of the input and output pair trigger array isn't identical in the general case
   * - :doc:`get_best_candidate_segments <../functions/coincidence_HM.get_best_candidate_segments>`
     - Returns best trigger pair given two lists of friends. Doesn't impose a time-delay constraint, assumes that time-delay is either irrelevant (bg), or taken care of by using the appropriate time_shift_tol
   * - :doc:`get_files_with_glitches <../functions/coincidence_HM.get_files_with_glitches>`
     - No docstring summary available.
   * - :doc:`get_friends_arr <../functions/coincidence_HM.get_friends_arr>`
     - Groups triggers in clist into buckets every time_tol, and retains only those high enough relative to the maximum in each bucket
   * - :doc:`get_histogram_stats <../functions/coincidence_HM.get_histogram_stats>`
     - No docstring summary available.
   * - :doc:`group_by_id <../functions/coincidence_HM.group_by_id>`
     - Groups the triggers in a clist into sublists with common calphas
   * - :doc:`inds_into_before <../functions/coincidence_HM.inds_into_before>`
     - Gives indices into cand_before_veto for triggers in cand_after_veto We can use tolist(), but this is almost 10 times faster
   * - :doc:`load_trigger_file <../functions/coincidence_HM.load_trigger_file>`
     - Convenience function to load triggers from a file
   * - :doc:`main <../functions/coincidence_HM.main>`
     - No docstring summary available.
   * - :doc:`mypad <../functions/coincidence_HM.mypad>`
     - No docstring summary available.
   * - :doc:`pick_lists <../functions/coincidence_HM.pick_lists>`
     - Returns elements from list1 whose corresponding elements in list2 aren't empty
   * - :doc:`read_channel_dict <../functions/coincidence_HM.read_channel_dict>`
     - Reads off boolean channel dict entries for times
   * - :doc:`remove_bad_times <../functions/coincidence_HM.remove_bad_times>`
     - No docstring summary available.
   * - :doc:`restart_run_from_json <../functions/coincidence_HM.restart_run_from_json>`
     - No docstring summary available.
   * - :doc:`secondary_peak_reject <../functions/coincidence_HM.secondary_peak_reject>`
     - Return True if this trigger should be rejected for having a secondary peak that ruins the coherent score
   * - :doc:`select_optimal_trigger_pairs <../functions/coincidence_HM.select_optimal_trigger_pairs>`
     - Function to return optimized background, with extra veto based on significant psd drift
   * - :doc:`stringent_veto <../functions/coincidence_HM.stringent_veto>`
     - Apply stringent vetoes to optimized candidates for the edge case in which the best one in the bucket survives, but the coincident ones do not
   * - :doc:`track_job <../functions/coincidence_HM.track_job>`
     - Track a multiprocessing job submitted in chunks
   * - :doc:`veto_and_optimize_coincidence_list <../functions/coincidence_HM.veto_and_optimize_coincidence_list>`
     - Returns list of vetoed and optimized coincident candidates (w/ timeslides) and any extra information
   * - :doc:`veto_and_optimize_group <../functions/coincidence_HM.veto_and_optimize_group>`
     - -
   * - :doc:`veto_and_optimize_single_detector <../functions/coincidence_HM.veto_and_optimize_single_detector>`
     - Veto and optimize coincident candidates in a single detector
   * - :doc:`veto_orthogonal_step <../functions/coincidence_HM.veto_orthogonal_step>`
     - -

No public classes were found.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../functions/coincidence_HM.collect_all_candidate_files_npy
   ../functions/coincidence_HM.collect_all_candidate_files_npz
   ../functions/coincidence_HM.collect_background_candidates
   ../functions/coincidence_HM.collect_files
   ../functions/coincidence_HM.collect_files_npz
   ../functions/coincidence_HM.combine_files
   ../functions/coincidence_HM.combine_files_npz
   ../functions/coincidence_HM.create_candidate_output_dir_name
   ../functions/coincidence_HM.create_shifted_observations
   ../functions/coincidence_HM.find_friends_in_cloud
   ../functions/coincidence_HM.find_interesting_dir
   ../functions/coincidence_HM.find_interesting_dir_cluster
   ../functions/coincidence_HM.fish_in_weaker_detector
   ../functions/coincidence_HM.fit_step_veto
   ../functions/coincidence_HM.flag_duplicates_per_group_pair
   ../functions/coincidence_HM.get_best_candidate_segments
   ../functions/coincidence_HM.get_files_with_glitches
   ../functions/coincidence_HM.get_friends_arr
   ../functions/coincidence_HM.get_histogram_stats
   ../functions/coincidence_HM.group_by_id
   ../functions/coincidence_HM.inds_into_before
   ../functions/coincidence_HM.load_trigger_file
   ../functions/coincidence_HM.main
   ../functions/coincidence_HM.mypad
   ../functions/coincidence_HM.pick_lists
   ../functions/coincidence_HM.read_channel_dict
   ../functions/coincidence_HM.remove_bad_times
   ../functions/coincidence_HM.restart_run_from_json
   ../functions/coincidence_HM.secondary_peak_reject
   ../functions/coincidence_HM.select_optimal_trigger_pairs
   ../functions/coincidence_HM.stringent_veto
   ../functions/coincidence_HM.track_job
   ../functions/coincidence_HM.veto_and_optimize_coincidence_list
   ../functions/coincidence_HM.veto_and_optimize_group
   ../functions/coincidence_HM.veto_and_optimize_single_detector
   ../functions/coincidence_HM.veto_orthogonal_step
