coherent_score_hm_search.compute_coherent_scores
================================================

Back to :doc:`Module page <../modules/coherent_score_hm_search>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def compute_coherent_scores(cs_instance, coincident_trig_list, trig1, trig2, minimal_time_slide_jump = 0.1, score_reduction_timeseries = 10, output_timeseries = False, output_coherent_score = True, coh_score_iterations = 1, return_inputs = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``cs_instance``
     - -
     - -
     - instance of SearchCoherentScoreHMAS
   * - ``coincident_trig_list``
     - -
     - -
     - list of coincident triggers
   * - ``trig1``
     - -
     - -
     - trigger object for detector 1
   * - ``trig2``
     - -
     - -
     - trigger object for detector 2
   * - ``minimal_time_slide_jump``
     - -
     - 0.1
     - minimal time slide jump
   * - ``score_reduction_timeseries``
     - -
     - 10
     - -
   * - ``output_timeseries``
     - -
     - False
     - -
   * - ``output_coherent_score``
     - -
     - True
     - -
   * - ``coh_score_iterations``
     - -
     - 1
     - to improve convergence, one could average over multiple iterations of the coherent score calculation
   * - ``return_inputs``
     - -
     - False
     - for debugging purposes, return the inputs to the coherent score calculation (only for the first trigger in the list)

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

   :param cs_instance: instance of SearchCoherentScoreHMAS
   :param coincident_trig_list: list of coincident triggers
   :param trig1: trigger object for detector 1
   :param trig2: trigger object for detector 2
   :param minimal_time_slide_jump: minimal time slide jump
   :param score_reduction_timeseries:
   :param output_timeseries:
   :param output_coherent_score:
   :param coh_score_iterations: to improve convergence, one could average
           over multiple iterations of the coherent score calculation
   :param return_inputs: for debugging purposes, return the inputs to the
           coherent score calculation (only for the first trigger in the list)
