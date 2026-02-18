ranking_HM.maximize_over_banks
==============================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Note that Seth vetoed the triggers before the bank assignment

Signature
---------

.. code-block:: python

   def maximize_over_banks(list_of_rank_objs, maxopts_filepath = None, incoherent_score_func = utils.incoherent_score, coherent_score_func = utils.coherent_score, mask_veto_criteria = None, apply_veto_before_scoring = False, global_maximization_format = 'new', matching_point = None, downsampling_correction = True, include_vetoed_triggers = False, p_veto_real_event = (DEFAULT_P_VETO, DEFAULT_P_VETO), **ranking_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``list_of_rank_objs``
     - -
     - -
     - List of Rank objects
   * - ``maxopts_filepath``
     - -
     - None
     - Path to a hdf5 file to save the maximization options to. If it doesn't exist, it's created and populated with the given options. If it exists, it's overwritten.
   * - ``incoherent_score_func``
     - -
     - utils.incoherent_score
     - Function that accepts two processedclists for a trigger and returns an incoherent score
   * - ``coherent_score_func``
     - -
     - utils.coherent_score
     - Function that accepts the coherent terms for a trigger and returns the coherent score (just the sum by default)
   * - ``mask_veto_criteria``
     - -
     - None
     - If needed, pass in a mask on rank_obj.veto_metadata_keys to identify glitch tests that we use, to override the default (everything)
   * - ``apply_veto_before_scoring``
     - -
     - False
     - Whether to apply the vetoes before scoring (we always apply them after scoring anyway) Can be a boolean variable, or a mask on rank_obj.veto_metadata_keys To reproduce O3a, i.e., 2201.02252, pass a mask with ones everywhere except at the entry corresponding to 'Secondary_peak_timeseries' False is the recommended input for all future catalogs
   * - ``global_maximization_format``
     - -
     - 'new'
     - 'new' for the new format, 'old' for the old format If 'new', we ensure that the maximization over banks/subbanks does not depend on the ordering 'old' was the default for all published catalogs before 08-29-2024 'new' is the recommended input for all future catalogs
   * - ``matching_point``
     - -
     - None
     - Where we match the rank functions
   * - ``downsampling_correction``
     - -
     - True
     - If the triggers were downsampled compared to a chi-sq distribution because of an additional cut (e.g., based on whether the mode ratios A33/A22 or A44/A22 are physical). This flag corrects the rank function so that it follows the chi-sq behavior again. This flag needs a file downsamp_corr_path.npy to be input when creating the Rank instance
   * - ``include_vetoed_triggers``
     - -
     - False
     - Flag whether to include the triggers which failed the vetos in our final list
   * - ``p_veto_real_event``
     - -
     - (DEFAULT_P_VETO, DEFAULT_P_VETO)
     - Tuple with functions for the probability that a real event fails the vetoes in each detector, which in the most general case can be a function of all properties of the trigger. The functions should accept a list of entries of scores_(non)vetoed_max and yield an array of probabilities. Used only if include_vetoed_triggers is True
   * - ``\*\*ranking_kwargs``
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
     - Considers all the banks in list_of_rank_objs together, assigns each trigger to a single (bank, subbank) pair and populates cands_preveto_max in all the banks

Docstring
---------

.. code-block:: text

   Note that Seth vetoed the triggers before the bank assignment
   :param list_of_rank_objs: List of Rank objects
   :param maxopts_filepath:
       Path to a hdf5 file to save the maximization options to.
       If it doesn't exist, it's created and populated with the given options.
       If it exists, it's overwritten.
   :param incoherent_score_func:
       Function that accepts two processedclists for a trigger and returns an
       incoherent score
   :param coherent_score_func:
       Function that accepts the coherent terms for a trigger and returns
       the coherent score (just the sum by default)
   :param mask_veto_criteria:
       If needed, pass in a mask on rank_obj.veto_metadata_keys to
       identify glitch tests that we use, to override the default (everything)
   :param apply_veto_before_scoring:
       Whether to apply the vetoes before scoring (we always apply them after
       scoring anyway)
       Can be a boolean variable, or a mask on rank_obj.veto_metadata_keys
       To reproduce O3a, i.e., 2201.02252, pass a mask with ones everywhere
       except at the entry corresponding to 'Secondary_peak_timeseries'
       False is the recommended input for all future catalogs
   :param global_maximization_format:
       'new' for the new format, 'old' for the old format
       If 'new', we ensure that the maximization over banks/subbanks does not
       depend on the ordering
       'old' was the default for all published catalogs before 08-29-2024
       'new' is the recommended input for all future catalogs
   :param downsampling_correction:
       If the triggers were downsampled compared to a chi-sq distribution
       because of an additional cut (e.g., based on whether the mode ratios
       A33/A22 or A44/A22 are physical). This flag corrects the rank function
       so that it follows the chi-sq behavior again. This flag needs a file
       downsamp_corr_path.npy to be input when creating the Rank instance
   :param matching_point: Where we match the rank functions
   :param include_vetoed_triggers: Flag whether to include the triggers which
       failed the vetos in our final list
   :param p_veto_real_event:
       Tuple with functions for the probability that a real event fails the
       vetoes in each detector, which in the most general case can be a
       function of all properties of the trigger. The functions should accept a
       list of entries of scores_(non)vetoed_max and yield an array of
       probabilities. Used only if include_vetoed_triggers is True
   :return:
       Considers all the banks in list_of_rank_objs together, assigns each
       trigger to a single (bank, subbank) pair and populates
       cands_preveto_max in all the banks
