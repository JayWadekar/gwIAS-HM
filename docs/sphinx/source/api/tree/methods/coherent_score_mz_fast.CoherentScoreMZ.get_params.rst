coherent_score_mz_fast.CoherentScoreMZ.get_params
=================================================

Back to :doc:`Class page <../classes/coherent_score_mz_fast.CoherentScoreMZ>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def get_params(self, events, time_slide_jump = DEFAULT_TIMESLIDE_JUMP / 1000)

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
   * - ``time_slide_jump``
     - -
     - DEFAULT_TIMESLIDE_JUMP / 1000
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
     - n_cand x (n_det = 2) x 4 array (always 3D) with shifted_ts, re(z), im(z), effective_sensitivity in each detector

Docstring
---------

.. code-block:: text

   :param events:
       (n_cand x (n_det = 2) x processedclist)/
       ((n_det = 2) x processedclist) array with coincidence/background
       candidates
   :param time_slide_jump: Units of jumps (s) for timeslides
   :return: n_cand x (n_det = 2) x 4 array (always 3D) with
       shifted_ts, re(z), im(z), effective_sensitivity in each detector
