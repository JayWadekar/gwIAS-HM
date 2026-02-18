ranking_HM.get_params
=====================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

TODO_Jay: We only need the sensitivity (where?) and not ts_out, rezs, imzs

Signature
---------

.. code-block:: python

   def get_params(events, clist_pos, time_slide_jump = params.DEFAULT_TIMESLIDE_JUMP / 1000)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``events``
     - -
     - -
     - (n_cand x (n_det = 2) x processedclist)/ ((n_det = 2) x processedclist) array with coincidence/background candidates
   * - ``clist_pos``
     - -
     - -
     - Dictionary with the name of trigger attributes as keys and the index of the attributes in the processedclist as values
   * - ``time_slide_jump``
     - -
     - params.DEFAULT_TIMESLIDE_JUMP / 1000
     - Units of jumps (s) for timeslides

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_cand x (n_det = 2) x 4 array (always 3D) with shifted_ts, re(rho_22), im(rho_22), effective_sensitivity in each detector

Docstring
---------

.. code-block:: text

   TODO_Jay: We only need the sensitivity (where?) and not ts_out, rezs, imzs
   :param events:
       (n_cand x (n_det = 2) x processedclist)/
       ((n_det = 2) x processedclist) array with coincidence/background
       candidates
   :param clist_pos:
       Dictionary with the name of trigger attributes as keys and the index of
       the attributes in the processedclist as values
   :param time_slide_jump: Units of jumps (s) for timeslides
   :return: n_cand x (n_det = 2) x 4 array (always 3D) with
       shifted_ts, re(rho_22), im(rho_22), effective_sensitivity in each detector
