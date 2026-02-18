ranking_HM.veto_rank_entry
==========================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def veto_rank_entry(rank_entry, ind, triggers_dir = None, source = 'BBH', runs = ('hm_o3a', 'hm_o3b'), min_veto_chi2 = 30, idx_veto_metadata = -2, detectors = ('H1', 'L1'), veto_inds = None, rerun_coh_score = False, cs_instance = None, time_slide_jump = params.DEFAULT_TIMESLIDE_JUMP / 1000.0, coh_score_iterations = 10, return_trig_obj_info = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``rank_entry``
     - -
     - -
     - Entry of one of the arrays in cands_(pre/post)veto_max
   * - ``ind``
     - -
     - -
     - Index into cands_preveto_max[cand_type], only used to debug which trigger failed
   * - ``triggers_dir``
     - -
     - None
     - Directory where the triggers are stored, exposed for debugging in case we're using a non-standard directory
   * - ``source``
     - -
     - 'BBH'
     - Should be one of 'BBH', 'BNS', or 'NSBH'
   * - ``runs``
     - -
     - ('hm_o3a', 'hm_o3b')
     - Iterable with runs being searched in
   * - ``min_veto_chi2``
     - -
     - 30
     - Minimum chi2 value to veto
   * - ``idx_veto_metadata``
     - -
     - -2
     - Index of veto_metadata in the rank_entry
   * - ``detectors``
     - -
     - ('H1', 'L1')
     - Iterable of detectors, should match the order in the processedclists
   * - ``veto_inds``
     - -
     - None
     - Indices of vetoes to apply (only used for check)
   * - ``rerun_coh_score``
     - -
     - False
     - Flag to rerun the coherent score computation (only for triggers that passed the vetoes)
   * - ``cs_instance``
     - -
     - None
     - Coherent score instance, used only if rerun_coh_score
   * - ``time_slide_jump``
     - -
     - params.DEFAULT_TIMESLIDE_JUMP / 1000.0
     - The least count of the timeslides
   * - ``coh_score_iterations``
     - -
     - 10
     - Number of iterations for the coherent score
   * - ``return_trig_obj_info``
     - -
     - False
     - Flag to return the trigger objects (only used for debugging)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - New veto metadata and coherent score

Docstring
---------

.. code-block:: text

   :param rank_entry: Entry of one of the arrays in cands_(pre/post)veto_max
   :param ind:
       Index into cands_preveto_max[cand_type], only used to debug which
       trigger failed
   :param triggers_dir:
       Directory where the triggers are stored, exposed for debugging in case
       we're using a non-standard directory
   :param source: Should be one of 'BBH', 'BNS', or 'NSBH'
   :param runs: Iterable with runs being searched in
   :param min_veto_chi2: Minimum chi2 value to veto
   :param idx_veto_metadata: Index of veto_metadata in the rank_entry
   :param detectors:
       Iterable of detectors, should match the order in the processedclists
   :param veto_inds: Indices of vetoes to apply (only used for check)
   :param rerun_coh_score:
       Flag to rerun the coherent score computation
       (only for triggers that passed the vetoes)
   :param cs_instance: Coherent score instance, used only if rerun_coh_score
   :param time_slide_jump: The least count of the timeslides
   :param coh_score_iterations: Number of iterations for the coherent score
   :param return_trig_obj_info:
       Flag to return the trigger objects (only used for debugging)
   :return: New veto metadata and coherent score
