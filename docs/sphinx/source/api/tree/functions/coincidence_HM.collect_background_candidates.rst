coincidence_HM.collect_background_candidates
============================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Function to collect background candidates (including real events)

Signature
---------

.. code-block:: python

   def collect_background_candidates(clist1, clist2, time_shift_tol, score_reduction_max, threshold_chi2, c0_pos, restricted_times = None, max_time_slide_shift = None, minimal_time_slide_jump = None, max_zero_lag_delay = None, origin = 0, score_func = utils.incoherent_score, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``clist1``
     - -
     - -
     - Processedclist 1
   * - ``clist2``
     - -
     - -
     - Processedclist 2
   * - ``time_shift_tol``
     - -
     - -
     - Width (s) of buckets to collect triggers into, the \\\`friends" of a trigger with the same calpha live within the same bucket
   * - ``score_reduction_max``
     - -
     - -
     - Absolute reduction in SNR^2 from the peak value in each bucket to retain (we also have a hardcoded relative reduction)
   * - ``threshold_chi2``
     - -
     - -
     - Threshold in sum(SNR^2) above which we consider triggers for the background list (or signal)
   * - ``c0_pos``
     - -
     - -
     - Index of c0 in the processedclists
   * - ``restricted_times``
     - -
     - None
     - If needed, list of times to restrict analysis to save time during injection campaign
   * - ``max_time_slide_shift``
     - -
     - None
     - Max delay allowed for background triggers
   * - ``minimal_time_slide_jump``
     - -
     - None
     - Jumps in timeslides
   * - ``max_zero_lag_delay``
     - -
     - None
     - Maximum delay between detectors within the same timeslide. If not given, it defaults to time_shift_tol
   * - ``origin``
     - -
     - 0
     - Origin for splitting the times relative to
   * - ``score_func``
     - -
     - utils.incoherent_score
     - Function that accepts coincident trigger(s) and returns score(s)
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
     - n_candidate x 2 x len(Processedclist[0]) array with candidates

Docstring
---------

.. code-block:: text

   Function to collect background candidates (including real events)
   :param clist1: Processedclist 1
   :param clist2: Processedclist 2
   :param time_shift_tol:
       Width (s) of buckets to collect triggers into, the `friends" of a
       trigger with the same calpha live within the same bucket
   :param score_reduction_max:
       Absolute reduction in SNR^2 from the peak value in each bucket to
       retain (we also have a hardcoded relative reduction)
   :param threshold_chi2:
       Threshold in sum(SNR^2) above which we consider triggers for the
       background list (or signal)
   :param c0_pos: Index of c0 in the processedclists
   :param restricted_times:
       If needed, list of times to restrict analysis to save time during
       injection campaign
   :param max_time_slide_shift: Max delay allowed for background triggers
   :param minimal_time_slide_jump: Jumps in timeslides
   :param max_zero_lag_delay:
       Maximum delay between detectors within the same timeslide.
       If not given, it defaults to time_shift_tol
   :param origin: Origin for splitting the times relative to
   :param score_func:
       Function that accepts coincident trigger(s) and returns score(s)
   :param kwargs: Dictionary with extra parameters for score_func
   :return: n_candidate x 2 x len(Processedclist[0]) array with candidates
