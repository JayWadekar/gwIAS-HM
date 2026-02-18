coincidence_HM.select_optimal_trigger_pairs
===========================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Function to return optimized background, with extra veto based on significant psd drift

Signature
---------

.. code-block:: python

   def select_optimal_trigger_pairs(candidates, veto_dicts, time_shift_tol, threshold_chi2, trig_obj, origin = 0, score_func = utils.incoherent_score, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``candidates``
     - -
     - -
     - n_candidates x 2 x len(processedclist[0]) array
   * - ``veto_dicts``
     - -
     - -
     - Dictionaries as output by veto_and_optimize_single_detector
   * - ``time_shift_tol``
     - -
     - -
     - Tolerance to sort candidates into buckets
   * - ``threshold_chi2``
     - -
     - -
     - Incoherent threshold (only for veto based on significant PSD drift)
   * - ``trig_obj``
     - -
     - -
     - Trigger object, used to read off an index offset, position of c0, and dimensions in the finer grid
   * - ``origin``
     - -
     - 0
     - Origin for splitting the times relative to
   * - ``score_func``
     - -
     - utils.incoherent_score
     - Function that accepts two triggers and returns a score
   * - ``\*\*kwargs``
     - -
     - -
     - Dictionary with extra parameters for score_func

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. List of optimized background candidates 2. Mask into candidates that passed all the vetoes 3. len(optimized_candidates) x 2 x (len(self.outlier_reasons) + 6 + len(split_chunks)) boolean array with metadata about the trigger in each detector (see veto_and_optimize_group for the meaning of the entries)

Docstring
---------

.. code-block:: text

   Function to return optimized background, with extra veto based on
   significant psd drift
   :param candidates: n_candidates x 2 x len(processedclist[0]) array
   :param veto_dicts:
       Dictionaries as output by veto_and_optimize_single_detector
   :param time_shift_tol: Tolerance to sort candidates into buckets
   :param threshold_chi2:
       Incoherent threshold (only for veto based on significant PSD drift)
   :param trig_obj:
       Trigger object, used to read off an index offset, position of c0, and 
       dimensions in the finer grid
   :param origin: Origin for splitting the times relative to
   :param score_func: Function that accepts two triggers and returns a score
   :param kwargs: Dictionary with extra parameters for score_func
   :return: 1. List of optimized background candidates
            2. Mask into candidates that passed all the vetoes
            3. len(optimized_candidates) x 2 x
               (len(self.outlier_reasons) + 6 + len(split_chunks))
               boolean array with metadata about the trigger in each detector
               (see veto_and_optimize_group for the meaning of the entries)
