ranking_HM.compute_coherent_scores
==================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

1. Computes coherent scores if not computed, and prior and sensitivity terms 2. Edits hdf5 file object fobj in place to include prior terms if applicable 3. Edits cands_by_subbank in place to include extra prior and sensitivity terms if coherent scores were saved 4.

Signature
---------

.. code-block:: python

   def compute_coherent_scores(cands_by_subbank, extra_array_names, clist_pos, time_slide_jump = 0.1, mask_veto_criteria = None, median_normfacs_by_subbank = None, template_prior_funcs = None, fobj = None, candtype = FOBJ_KEYS[0])

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``cands_by_subbank``
     - -
     - -
     - An element of bg_fg_by_subbank - a dictionary with loc_id as keys, with n_cand x (n_det = 2) x row of processedclist arrays with triggers in the subbank (also w/ timeseries, veto_metadata, coherent scores if we have them)
   * - ``extra_array_names``
     - -
     - -
     - List of names of the extra arrays in bg_by_subbank
   * - ``clist_pos``
     - -
     - -
     - Dictionary with the name of trigger attributes as keys and the index of the attributes in the processedclist as values
   * - ``time_slide_jump``
     - -
     - 0.1
     - The least-count of timeslides (s)
   * - ``mask_veto_criteria``
     - -
     - None
     - If needed, mask on veto_metadata.shape[-1] that we use to pick criteria to veto candidates on (used only if veto_metadata is available)
   * - ``median_normfacs_by_subbank``
     - -
     - None
     - If known, dictionary with loc_id as keys, and n_det array of median normfacs as values. If None, we estimate them from the data (should only estimate for the background!)
   * - ``template_prior_funcs``
     - -
     - None
     - If known, dictionary indexed by subbank_id, with (function that returns template prior given calpha, # input dimensions)
   * - ``fobj``
     - -
     - None
     - If bg_by_subbank is read from a hdf5 file, the File object (must be writeable)
   * - ``candtype``
     - -
     - FOBJ_KEYS[0]
     - Entry into FOBJ_KEYS for the type of candidates

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. List of lists (n_subbank x 4 + n_extra_arrays - 1 (if prior is within) + 1) with each subbank's contribution to the lists in scores_(non)vetoed_max (for the background, this is scores_bg_by_subbank_nonvetoed) Each entry of a subbank's list is composed of a) The prior terms (coherent score, -rho^2, 2 log(1/median normfac^3)), 2 log(template prior)) b) (n_det=2) x row of processedclist array with the trigger c) (bank_id, subbank_id) b) 4 x (n_det=2) array with t, Re(rho_22), Im(rho_22), sensitivity ratio e)... extra info that was passed in about each trigger minus the coherent score which is already in the prior terms f) The index of the trigger into bg_by_subbank, useful as things can be reordered when we maximize over banks 2. Dictionary giving median H1 and L1 normfacs, with subbank_ids as keys (median_normfacs_by_subbank)

Docstring
---------

.. code-block:: text

   1. Computes coherent scores if not computed, and prior and sensitivity terms
   2. Edits hdf5 file object fobj in place to include prior terms if applicable
   3. Edits cands_by_subbank in place to include extra prior and sensitivity
      terms if coherent scores were saved
   4. Returns entries like those of scores_bg_by_subbank_nonvetoed (for fg,
      combine outside this function) and median_normfacs_by_subbank
   :param cands_by_subbank:
       An element of bg_fg_by_subbank - a dictionary with loc_id as keys, with
       n_cand x (n_det = 2) x row of processedclist arrays with triggers in the
       subbank (also w/ timeseries, veto_metadata, coherent scores if we have
       them)
   :param extra_array_names: List of names of the extra arrays in bg_by_subbank
   :param clist_pos:
       Dictionary with the name of trigger attributes as keys and the index of
       the attributes in the processedclist as values
   :param time_slide_jump: The least-count of timeslides (s)
   :param mask_veto_criteria:
       If needed, mask on veto_metadata.shape[-1] that we use to pick criteria
       to veto candidates on (used only if veto_metadata is available)
   :param median_normfacs_by_subbank:
       If known, dictionary with loc_id as keys, and n_det array of median
       normfacs as values. If None, we estimate them from the data (should only
       estimate for the background!)
   :param template_prior_funcs:
       If known, dictionary indexed by subbank_id, with
       (function that returns template prior given calpha, # input dimensions)
   :param fobj:
       If bg_by_subbank is read from a hdf5 file, the File object (must be
       writeable)
   :param candtype: Entry into FOBJ_KEYS for the type of candidates
   :return: 1. List of lists (n_subbank x
                4 + n_extra_arrays - 1 (if prior is within) + 1) with each
               subbank's contribution to the lists in scores_(non)vetoed_max
               (for the background, this is scores_bg_by_subbank_nonvetoed)
               Each entry of a subbank's list is composed of
                   a) The prior terms
                       (coherent score,
                       -rho^2,
                       2 log(1/median normfac^3)),
                       2 log(template prior))
                   b) (n_det=2) x row of processedclist array with the trigger
                   c) (bank_id, subbank_id)
                   b) 4 x (n_det=2) array with t, Re(rho_22), Im(rho_22), sensitivity ratio
                   e)... extra info that was passed in about each trigger minus
                      the coherent score which is already in the prior terms
                   f) The index of the trigger into bg_by_subbank, useful as
                      things can be reordered when we maximize over banks
            2. Dictionary giving median H1 and L1 normfacs, with
               subbank_ids as keys (median_normfacs_by_subbank)
