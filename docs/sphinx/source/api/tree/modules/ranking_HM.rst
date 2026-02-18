ranking_HM
==========

Back to :doc:`API tree index <../index>`

Purpose
-------

Candidate aggregation, reweighting, and ranking logic.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`add_or_vstack_to_dic <../functions/ranking_HM.add_or_vstack_to_dic>`
     - No docstring summary available.
   * - :doc:`add_to_bg_and_fg_dicts <../functions/ranking_HM.add_to_bg_and_fg_dicts>`
     - Populates events in lists of pure background, zero-lag candidates, LSC events, and injected events
   * - :doc:`apply_vetos <../functions/ranking_HM.apply_vetos>`
     - No docstring summary available.
   * - :doc:`apply_vetos_det <../functions/ranking_HM.apply_vetos_det>`
     - No docstring summary available.
   * - :doc:`check_and_fix_bg_fg_lists <../functions/ranking_HM.check_and_fix_bg_fg_lists>`
     - Our injection codes populate bg_fg_max, but not bg_fg_by_subbank. We also want to save indices into bg_fg_by_subbank in bg_fg_max to avoid duplication in memory, but old codes might not have saved it in this format. This function fixes both issues. Run this before scoring, and re-score after.
   * - :doc:`collect_all_subbanks <../functions/ranking_HM.collect_all_subbanks>`
     - No docstring summary available.
   * - :doc:`combine_subbank_dict <../functions/ranking_HM.combine_subbank_dict>`
     - Populates a single list in bg_fg_max
   * - :doc:`compute_coherent_scores <../functions/ranking_HM.compute_coherent_scores>`
     - 1. Computes coherent scores if not computed, and prior and sensitivity terms 2. Edits hdf5 file object fobj in place to include prior terms if applicable 3. Edits cands_by_subbank in place to include extra prior and sensitivity terms if coherent scores were saved 4.
   * - :doc:`correct_snr2sumcut_safe <../functions/ranking_HM.correct_snr2sumcut_safe>`
     - No docstring summary available.
   * - :doc:`generate_downsampling_correction_fits <../functions/ranking_HM.generate_downsampling_correction_fits>`
     - We want to calculate the non-Gaussian correction to the ranking statistic (see arXiv: 2405.17400). We therefore want to compare the empirical histogram of the SNR^2 to that expected from the Gaussian noise hypothesis (see Fig.3 of the paper). The empirical histogram is unfortunately also affected by downsampling at the low SNR^2 end due to using HM marginalized statistic based threshold in the matched_filtering stage (instead of using rho_incoherent^2 as the threshold). To remedy this, we do a quick and rough calculation of the downsampling correction. and fit it by a parabolic function to avoid noise at the high SNR^2 end. The fit is then used to correct the empirical histogram or rank function.
   * - :doc:`get_params <../functions/ranking_HM.get_params>`
     - TODO_Jay: We only need the sensitivity (where?) and not ts_out, rezs, imzs
   * - :doc:`maximization_info_from_pclists <../functions/ranking_HM.maximization_info_from_pclists>`
     - Returns information useful for maximize_over_groups
   * - :doc:`maximize_over_banks <../functions/ranking_HM.maximize_over_banks>`
     - Note that Seth vetoed the triggers before the bank assignment
   * - :doc:`maximize_over_groups <../functions/ranking_HM.maximize_over_groups>`
     - -
   * - :doc:`maximize_using_saved_options <../functions/ranking_HM.maximize_using_saved_options>`
     - Convenience function to use saved options to maximize_over_banks
   * - :doc:`offset_background <../functions/ranking_HM.offset_background>`
     - Finds the amount to shift the detectors' data streams by
   * - :doc:`parabolic_func <../functions/ranking_HM.parabolic_func>`
     - Define a parabolic function to fit downsampling corrections
   * - :doc:`pick_mask <../functions/ranking_HM.pick_mask>`
     - No docstring summary available.
   * - :doc:`print_top_cand_list <../functions/ranking_HM.print_top_cand_list>`
     - Print the top candidates from the rank_objs
   * - :doc:`reform_coherent_scores <../functions/ranking_HM.reform_coherent_scores>`
     - Edits input structures in place via aliasing and returns scores_bg_by_subbank
   * - :doc:`split_into_subbank_dicts <../functions/ranking_HM.split_into_subbank_dicts>`
     - -
   * - :doc:`veto_rank_entry <../functions/ranking_HM.veto_rank_entry>`
     - -
   * - :doc:`veto_rank_entry_wrapper <../functions/ranking_HM.veto_rank_entry_wrapper>`
     - No docstring summary available.
   * - :doc:`veto_subthreshold_top_cands <../functions/ranking_HM.veto_subthreshold_top_cands>`
     - Top candidates in the cands_preveto_max list which are subthreshold (i.e. have SNR below min_veto_chi2) are passed through all the veto tests again

.. list-table:: Classes
   :header-rows: 1

   * - Class
     - Summary
   * - :doc:`Rank <../classes/ranking_HM.Rank>`
     - No class docstring summary available.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../classes/ranking_HM.Rank
   ../functions/ranking_HM.add_or_vstack_to_dic
   ../functions/ranking_HM.add_to_bg_and_fg_dicts
   ../functions/ranking_HM.apply_vetos
   ../functions/ranking_HM.apply_vetos_det
   ../functions/ranking_HM.check_and_fix_bg_fg_lists
   ../functions/ranking_HM.collect_all_subbanks
   ../functions/ranking_HM.combine_subbank_dict
   ../functions/ranking_HM.compute_coherent_scores
   ../functions/ranking_HM.correct_snr2sumcut_safe
   ../functions/ranking_HM.generate_downsampling_correction_fits
   ../functions/ranking_HM.get_params
   ../functions/ranking_HM.maximization_info_from_pclists
   ../functions/ranking_HM.maximize_over_banks
   ../functions/ranking_HM.maximize_over_groups
   ../functions/ranking_HM.maximize_using_saved_options
   ../functions/ranking_HM.offset_background
   ../functions/ranking_HM.parabolic_func
   ../functions/ranking_HM.pick_mask
   ../functions/ranking_HM.print_top_cand_list
   ../functions/ranking_HM.reform_coherent_scores
   ../functions/ranking_HM.split_into_subbank_dicts
   ../functions/ranking_HM.veto_rank_entry
   ../functions/ranking_HM.veto_rank_entry_wrapper
   ../functions/ranking_HM.veto_subthreshold_top_cands
