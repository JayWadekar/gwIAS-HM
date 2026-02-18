ranking_HM.Rank.score_bg_fg_lists
=================================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

Compute quantities needed to assign scores to the triggers Responsible for populating: 1. coherent scores 2. rank functions (CDF of P(SNR^2\\\|H0) in each detector) If we are rerunning with a different subset of vetoes, update the desired set in self.mask_veto_criteria and, if apply_veto_before_scoring is True, set redo_bg = True

Signature
---------

.. code-block:: python

   def score_bg_fg_lists(self, redo_bg = False, redo_fg = False, apply_veto_before_scoring = True, score_triggers = True, coherent_score_func = utils.coherent_score, include_vetoed_triggers = False, safety_factor = 4, matching_point = None, scoring_method = 'old', downsampling_correction = True, min_trigs_per_grp = 500, p_veto_real_event = (DEFAULT_P_VETO, DEFAULT_P_VETO), **ranking_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``redo_bg``
     - -
     - False
     - Flag whether to override the saved coherent scores for background triggers and recompute them (also recomputes for the foreground)
   * - ``redo_fg``
     - -
     - False
     - Flag whether to override the saved coherent scores for foreground triggers and recompute them
   * - ``apply_veto_before_scoring``
     - -
     - True
     - Flag whether to apply the vetoes before scoring. The vetoes to apply are set by self.mask_veto_criteria on self.veto_metadata_keys (we apply them after scoring anyway)
   * - ``score_triggers``
     - -
     - True
     - Boolean flag to compute the scores
   * - ``coherent_score_func``
     - -
     - utils.coherent_score
     - Function that accepts the coherent terms for a trigger and returns the coherent score
   * - ``include_vetoed_triggers``
     - -
     - False
     - Flag to include vetoed triggers with a penalty in the final list
   * - ``safety_factor``
     - -
     - 4
     - Add this to threshold_chi2 before estimating the rank functions to account for incompleteness related to optimization
   * - ``matching_point``
     - -
     - None
     - Set SNR^2 at which the rank functions are matched, the default is threshold network SNR^2/2
   * - ``scoring_method``
     - -
     - 'old'
     - Flag to indicate whether we're using the old (mz/cdf) way of ranking vs new way (fitting for the pdf)
   * - ``downsampling_correction``
     - -
     - True
     - If the triggers were downsampled compared to a chi-sq distribution because of an additional cut (e.g., based on whether the mode ratios A33/A22 or A44/A22 are physical). This flag corrects the rank function so that it follows the chi-sq behavior again. This flag needs a file downsamp_corr_path.npy to be input when creating Rank class object
   * - ``min_trigs_per_grp``
     - -
     - 500
     - To avoid pathologies with making the rank functions, we require that the templates in each group have more than a particular number of background triggers associated to them
   * - ``p_veto_real_event``
     - -
     - (DEFAULT_P_VETO, DEFAULT_P_VETO)
     - Tuple with functions for the probability that a real event fails the vetoes in each detector, which in the most general case can be a function of all properties of the trigger. They should accept a list of entries of scores_(non)vetoed_max and yield an array of probabilities.
   * - ``\*\*ranking_kwargs``
     - -
     - -
     - Any extra arguments we want to pass to the ranking function

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

   Compute quantities needed to assign scores to the triggers
   Responsible for populating:
    1. coherent scores
    2. rank functions (CDF of P(SNR^2|H0) in each detector)
   If we are rerunning with a different subset of vetoes, update the
   desired set in self.mask_veto_criteria and, if
   apply_veto_before_scoring is True, set redo_bg = True
   :param redo_bg:
       Flag whether to override the saved coherent scores for background
       triggers and recompute them (also recomputes for the foreground)
   :param redo_fg:
       Flag whether to override the saved coherent scores for foreground
       triggers and recompute them
   :param apply_veto_before_scoring:
       Flag whether to apply the vetoes before scoring. The vetoes to
       apply are set by self.mask_veto_criteria on self.veto_metadata_keys
       (we apply them after scoring anyway)
   :param score_triggers: Boolean flag to compute the scores
   :param coherent_score_func:
       Function that accepts the coherent terms for a trigger and returns
       the coherent score
   :param include_vetoed_triggers:
       Flag to include vetoed triggers with a penalty in the final list
   :param safety_factor:
       Add this to threshold_chi2 before estimating the rank functions
       to account for incompleteness related to optimization
   :param matching_point:
       Set SNR^2 at which the rank functions are matched, the default
       is threshold network SNR^2/2
   :param scoring_method:
       Flag to indicate whether we're using the old (mz/cdf) way of ranking
       vs new way (fitting for the pdf)
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
   :param p_veto_real_event:
       Tuple with functions for the probability that a real event fails the
       vetoes in each detector, which in the most general case can be a
       function of all properties of the trigger. They should accept a list
       of entries of scores_(non)vetoed_max and yield an array of
       probabilities.
   :param ranking_kwargs:
       Any extra arguments we want to pass to the ranking function
   :return:
