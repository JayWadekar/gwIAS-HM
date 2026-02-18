ranking_HM.Rank.rank_scores_calc_MZ
===================================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

Calculating ranking scores for all the bg, fg, lsc, inj triggers. The rank functions are constructed in a sub-function make_score_funcs_MZ(). We first group the templates based on their "glitchiness" and then make separate rank functions for different groups

Signature
---------

.. code-block:: python

   def rank_scores_calc_MZ(self, i_subbank, safety_factor = 4, matching_point = None, downsampling_correction = True, min_trigs_per_grp = 500, output_rank_func_temp_grps = False, group_coarse_calphas = False, n_calpha_dim = 2, snr2_ref = None, n_snr2_ref = 0.001, vetoed = True, rhosqs_arrays = None, calphas_arrays = None, masks_vetoed = None, **score_func_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``i_subbank``
     - -
     - -
     - index of the subbank of the bank
   * - ``safety_factor``
     - -
     - 4
     - Add this to threshold_chi2 before estimating the rank functions to account for incompleteness related to optimization
   * - ``matching_point``
     - -
     - None
     - Set SNR^2 at which the rank functions are matched, the default is threshold network SNR^2/2 (the rank fns are made such that all template groups within the same subbank match at self.snr2min and the least glitchy template group in all subbanks match at matching_point)
   * - ``downsampling_correction``
     - -
     - True
     - If the triggers were downsampled compared to a chi-sq distribution because of an additional cut (e.g., based on whether the mode ratios A33/A22 or A44/A22 are physical). This flag corrects the rank function so that it follows the chi-sq behavior again. This flag needs a file downsamp_corr_path.npy to be input when creating Rank class object
   * - ``min_trigs_per_grp``
     - -
     - 500
     - To avoid pathologies with making the rank functions, we require that the templates in each group have more than a particular number of background triggers associated to them
   * - ``output_rank_func_temp_grps``
     - -
     - False
     - Flag if you want to output the rank funcs for separate template groups for reference
   * - ``group_coarse_calphas``
     - -
     - False
     - Flag whether we group the coarse or fine calphas
   * - ``n_calpha_dim``
     - -
     - 2
     - Number of dimensions to use for grouping calphas
   * - ``snr2_ref``
     - -
     - None
     - Quantify templates' glitchiness by the fraction of triggers with SNR>snr2_ref
   * - ``n_snr2_ref``
     - -
     - 0.001
     - Set snr2_ref by demanding that in the Gaussian noise case, the expected number of triggers > snr2_ref is n_snr2_ref over all calphas. Used only if snr2_ref is None
   * - ``vetoed``
     - -
     - True
     - Flag to compute scores only for vetoed triggers
   * - ``rhosqs_arrays``
     - -
     - None
     - If known, 4 x n_event x 2 array of SNR^2 values for background, foreground, LVC, injections (pass to avoid looping over hdf5)
   * - ``calphas_arrays``
     - -
     - None
     - If known, 4 x n_event x n_calpha_dim array of calphas
   * - ``masks_vetoed``
     - -
     - None
     - If known, array of length 4 with masks for vetoed triggers This makes us set vetoed to True
   * - ``\*\*score_func_kwargs``
     - -
     - -
     - -

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

   Calculating ranking scores for all the bg, fg, lsc, inj triggers.
   The rank functions are constructed in a sub-function
   make_score_funcs_MZ(). We first group the templates based on their
   "glitchiness" and then make separate rank functions for different groups
   :param i_subbank: index of the subbank of the bank
   :param safety_factor:
       Add this to threshold_chi2 before estimating the rank functions
       to account for incompleteness related to optimization
   :param matching_point:
       Set SNR^2 at which the rank functions are matched, the default
       is threshold network SNR^2/2
       (the rank fns are made such that all template groups within the same
        subbank match at self.snr2min and the least glitchy template group
        in all subbanks match at matching_point)
   :param downsampling_correction:
       If the triggers were downsampled compared to a chi-sq distribution
       because of an additional cut (e.g., based on whether the mode ratios 
       A33/A22 or A44/A22 are physical). This flag corrects the rank 
       function so that it follows the chi-sq behavior again. This flag
       needs a file downsamp_corr_path.npy to be input when creating
       Rank class object
   :param min_trigs_per_grp:
       To avoid pathologies with making the rank functions, we require that
       the templates in each group have more than a particular
       number of background triggers associated to them
   :param output_rank_func_temp_grps:
       Flag if you want to output the rank funcs for separate template groups
       for reference
   :param group_coarse_calphas:
       Flag whether we group the coarse or fine calphas
   :param n_calpha_dim: Number of dimensions to use for grouping calphas
   :param snr2_ref:
       Quantify templates' glitchiness by the fraction of triggers with
       SNR>snr2_ref
   :param n_snr2_ref:
       Set snr2_ref by demanding that in the Gaussian noise case, the
       expected number of triggers > snr2_ref is n_snr2_ref over all
       calphas. Used only if snr2_ref is None
   :param vetoed: Flag to compute scores only for vetoed triggers
   :param rhosqs_arrays:
       If known, 4 x n_event x 2 array of SNR^2 values for background,
       foreground, LVC, injections (pass to avoid looping over hdf5)
   :param calphas_arrays:
       If known, 4 x n_event x n_calpha_dim array of calphas
   :param masks_vetoed:
       If known, array of length 4 with masks for vetoed triggers
       This makes us set vetoed to True
