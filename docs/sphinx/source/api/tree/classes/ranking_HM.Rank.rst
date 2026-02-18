ranking_HM.Rank
===============

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

No class docstring summary available.

.. list-table:: Methods
   :header-rows: 1

   * - Method
     - Summary
   * - :doc:`from_pickle <../methods/ranking_HM.Rank.from_pickle>`
     - No docstring summary available.
   * - :doc:`to_pickle <../methods/ranking_HM.Rank.to_pickle>`
     - Deprecated on 6/2024 Saves the class to a pickle
   * - :doc:`to_hdf5 <../methods/ranking_HM.Rank.to_hdf5>`
     - Saves the class to a hdf5 file. Ensure that you ran check_and_fix_bg_fg_lists if you added extra injections (scoring does it automatically) Note that saving to a new path doesn't replace what is in the instance unless path is None, if you want to use the new hdf5 file created, load it using from_hdf5
   * - :doc:`from_hdf5 <../methods/ranking_HM.Rank.from_hdf5>`
     - Load the class from a hdf5 file
   * - :doc:`from_hdf5_static <../methods/ranking_HM.Rank.from_hdf5_static>`
     - Load attributes from a hdf5 file into an instance of the class
   * - :doc:`create_coh_score_instance <../methods/ranking_HM.Rank.create_coh_score_instance>`
     - Removed from init to make saved rank objects independent of the cogwheel version
   * - :doc:`reform_scores_lists <../methods/ranking_HM.Rank.reform_scores_lists>`
     - After maximizing over banks, there are fewer entries in cands_preveto_max, which are no longer in order of subbank_id. Apart from the return, this function 1. Recomputes the median normfacs for the subbanks 2. Reevaluates the sensitivity penalties for all candidates 3. Reorders the entries in cands_preveto_max
   * - :doc:`compute_function_on_subbank_data <../methods/ranking_HM.Rank.compute_function_on_subbank_data>`
     - It can be slow to loop over the entries in scores_(non)vetoed_max, read them into memory, and compute something, in bulk. It is faster to dereference the hdf5 arrays in bg_fg_by_subbank once, compute what we need and read the values relevant to scores_(non)vetoed_max
   * - :doc:`score_bg_fg_lists <../methods/ranking_HM.Rank.score_bg_fg_lists>`
     - Compute quantities needed to assign scores to the triggers Responsible for populating: 1. coherent scores 2. rank functions (CDF of P(SNR^2\\\|H0) in each detector) If we are rerunning with a different subset of vetoes, update the desired set in self.mask_veto_criteria and, if apply_veto_before_scoring is True, set redo_bg = True
   * - :doc:`rank_scores_calc <../methods/ranking_HM.Rank.rank_scores_calc>`
     - TODO: Speed this function up like rank_scores_calc_MZ Calculating ranking scores for all the bg, fg, lsc, inj triggers. We first group the templates based on their "glitchiness" and then make separate rank functions for different groups.
   * - :doc:`compute_fars_rank_score <../methods/ranking_HM.Rank.compute_fars_rank_score>`
     - Compute FARs using ranking scores as test statistics
   * - :doc:`compute_fars_coherent_score <../methods/ranking_HM.Rank.compute_fars_coherent_score>`
     - No docstring summary available.
   * - :doc:`avg_sensitive_volume <../methods/ranking_HM.Rank.avg_sensitive_volume>`
     - Run after scoring the bank bg
   * - :doc:`rank_scores_calc_MZ <../methods/ranking_HM.Rank.rank_scores_calc_MZ>`
     - Calculating ranking scores for all the bg, fg, lsc, inj triggers. The rank functions are constructed in a sub-function make_score_funcs_MZ(). We first group the templates based on their "glitchiness" and then make separate rank functions for different groups

Class docstring
---------------

No docstring text available.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../methods/ranking_HM.Rank.from_pickle
   ../methods/ranking_HM.Rank.to_pickle
   ../methods/ranking_HM.Rank.to_hdf5
   ../methods/ranking_HM.Rank.from_hdf5
   ../methods/ranking_HM.Rank.from_hdf5_static
   ../methods/ranking_HM.Rank.create_coh_score_instance
   ../methods/ranking_HM.Rank.reform_scores_lists
   ../methods/ranking_HM.Rank.compute_function_on_subbank_data
   ../methods/ranking_HM.Rank.score_bg_fg_lists
   ../methods/ranking_HM.Rank.rank_scores_calc
   ../methods/ranking_HM.Rank.compute_fars_rank_score
   ../methods/ranking_HM.Rank.compute_fars_coherent_score
   ../methods/ranking_HM.Rank.avg_sensitive_volume
   ../methods/ranking_HM.Rank.rank_scores_calc_MZ
