ranking_HM.Rank.rank_scores_calc
================================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

TODO: Speed this function up like rank_scores_calc_MZ Calculating ranking scores for all the bg, fg, lsc, inj triggers. We first group the templates based on their "glitchiness" and then make separate rank functions for different groups.

Signature
---------

.. code-block:: python

   def rank_scores_calc(self, i_subbank, safety_factor = 4, matching_point = None, output_rank_func_temp_grps = False, snr2_ref = 50, n_glitchy_groups_seed = 20, min_trigs_per_grp = 500, downsampling_correction = False, n_calpha_dim = 2, vetoed = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``i_subbank``
     - -
     - -
     - index of the subbank within self.scores_bg_by_subbank
   * - ``safety_factor``
     - -
     - 4
     - Add this to threshold_chi2 before estimating the rank functions to account for incompleteness related to optimization
   * - ``matching_point``
     - -
     - None
     - Set SNR^2 at which the rank functions are matched, the default is threshold network SNR^2/2 (the rank fns are made such that all template groups within the same subbank match at self.snr2min and the least glitchy template group in all subbanks match at matching_point)
   * - ``output_rank_func_temp_grps``
     - -
     - False
     - Flag if you want to output the rank funcs for separate template groups for reference (typically used for debugging)
   * - ``snr2_ref``
     - -
     - 50
     - Quantify templates' glitchiness by the fraction of triggers with SNR>snr2_ref
   * - ``n_glitchy_groups_seed``
     - -
     - 20
     - Number of seed groups to classify the templates into based on their glitchiness
   * - ``min_trigs_per_grp``
     - -
     - 500
     - To avoid pathologies with making the rank functions, we require that the templates in each group have more than a particular number of background triggers associated to them
   * - ``downsampling_correction``
     - -
     - False
     - If the triggers were downsampled compared to a chi-sq distribution because of an additional cut (e.g., based on whether the mode ratios A33/A22 or A44/A22 are physical). This flag corrects the rank function so that it follows the chi-sq behavior again. This flag needs a file downsamp_corr_path.npy to be input when creating Rank class object
   * - ``n_calpha_dim``
     - -
     - 2
     - Number of calpha dimensions to include while grouping triggers, some banks have extra dimensions that are out of control
   * - ``vetoed``
     - -
     - True
     - Flag to indicate whether we're computing the rank scores for vetoed or non-vetoed triggers

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - -

Docstring
---------

.. code-block:: text

   TODO: Speed this function up like rank_scores_calc_MZ
   Calculating ranking scores for all the bg, fg, lsc, inj triggers.
   We first group the templates based on their "glitchiness" and then make
   separate rank functions for different groups.
   :param i_subbank: index of the subbank within self.scores_bg_by_subbank
   :param safety_factor:
       Add this to threshold_chi2 before estimating the rank functions
       to account for incompleteness related to optimization
   :param matching_point:
       Set SNR^2 at which the rank functions are matched, the default
       is threshold network SNR^2/2
       (the rank fns are made such that all template groups within the same
        subbank match at self.snr2min and the least glitchy template group
        in all subbanks match at matching_point)
   :param output_rank_func_temp_grps:
       Flag if you want to output the rank funcs for separate template groups
       for reference (typically used for debugging)
   :param snr2_ref:
       Quantify templates' glitchiness by the fraction of triggers with
       SNR>snr2_ref
   :param n_glitchy_groups_seed:
       Number of seed groups to classify the templates into based on
       their glitchiness
   :param min_trigs_per_grp:
       To avoid pathologies with making the rank functions, we require that
       the templates in each group have more than a particular
       number of background triggers associated to them
   :param downsampling_correction:
       If the triggers were downsampled compared to a chi-sq distribution
       because of an additional cut (e.g., based on whether
       the mode ratios A33/A22 or A44/A22 are physical). This flag corrects
       the rank function so that it follows the chi-sq behavior again. This
       flag needs a file downsamp_corr_path.npy to be input when creating
       Rank class object
   :param n_calpha_dim:
       Number of calpha dimensions to include while grouping triggers,
       some banks have extra dimensions that are out of control
   :param vetoed: Flag to indicate whether we're computing the rank scores
       for vetoed or non-vetoed triggers
