coherent_score_mz_fast.CoherentScoreMZ.get_all_prior_terms
==========================================================

Back to :doc:`Class page <../classes/coherent_score_mz_fast.CoherentScoreMZ>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def get_all_prior_terms(self, events, timeseries = None, loc_id = None, ref_normfac = 1, time_slide_jump = DEFAULT_TIMESLIDE_JUMP / 1000, **score_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``events``
     - -
     - -
     - (n_cand x (n_det=2) x processedclist)/((n_det=2) x processedclist) array with coincidence/background candidates
   * - ``timeseries``
     - -
     - None
     - If known, list of lists/tuples of length n_detectors with n_samp x 3 array with t, Re(z), Im(z) (can be single list/tuple if n_events=1)
   * - ``loc_id``
     - -
     - None
     - Tuple with (bank_id, subbank_id), used if we're reading files
   * - ``ref_normfac``
     - -
     - 1
     - Reference normfac to scale the values relative to
   * - ``time_slide_jump``
     - -
     - DEFAULT_TIMESLIDE_JUMP / 1000
     - Least count of time slides (s)
   * - ``\*\*score_kwargs``
     - -
     - -
     - Extra arguments to trigger2comblist or comblist2cs

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Computes the coherent score minus the rho^2 piece

Docstring
---------

.. code-block:: text

   :param events:
       (n_cand x (n_det=2) x processedclist)/((n_det=2) x processedclist)
       array with coincidence/background candidates
   :param timeseries: If known, list of lists/tuples of length n_detectors
       with n_samp x 3 array with t, Re(z), Im(z)
       (can be single list/tuple if n_events=1)
   :param loc_id:
       Tuple with (bank_id, subbank_id), used if we're reading files
   :param ref_normfac: Reference normfac to scale the values relative to
   :param time_slide_jump: Least count of time slides (s)
   :param score_kwargs: Extra arguments to trigger2comblist or comblist2cs
   :return: Computes the coherent score minus the rho^2 piece
