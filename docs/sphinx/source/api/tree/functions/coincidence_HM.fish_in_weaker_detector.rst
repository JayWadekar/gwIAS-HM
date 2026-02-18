coincidence_HM.fish_in_weaker_detector
======================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Warning: This function has not been updated by Jay for HM case

Signature
---------

.. code-block:: python

   def fish_in_weaker_detector(trig_file_h1, trig_file_l1, trigger = None, time_lf = None, det_ind = 1, max_time_slide_shift = 100, time_shift_tol = 0.01, score_reduction_max = 5, minimal_time_slide_jump = 0.1, score_func = utils.incoherent_score, verbose = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trig_file_h1``
     - -
     - -
     - Trigger file with H1 triggers
   * - ``trig_file_l1``
     - -
     - -
     - Trigger file with L1 triggers
   * - ``trigger``
     - -
     - None
     - Row of processedclist with strong trigger, only used to fix the time
   * - ``time_lf``
     - -
     - None
     - Linear-free time, used if trigger wasn't given
   * - ``det_ind``
     - -
     - 1
     - Index of strong detector (0 for H1, 1 for L1)
   * - ``max_time_slide_shift``
     - -
     - 100
     - Max delay allowed for background triggers
   * - ``time_shift_tol``
     - -
     - 0.01
     - Maximum physical delay for zero-lag triggers
   * - ``score_reduction_max``
     - -
     - 5
     - Allow a reduction in SNR^2 of max \\\* params.MAX_FRIEND_DEGRADE_SNR2 + score_reduction_max when keeping friends
   * - ``minimal_time_slide_jump``
     - -
     - 0.1
     - Jumps in timeslides
   * - ``score_func``
     - -
     - utils.incoherent_score
     - Function to take two triggers and return the coherent score
   * - ``verbose``
     - -
     - False
     - Flag to print progress

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Background and zero-lag events as in coincidence, without cut

Docstring
---------

.. code-block:: text

   Warning: This function has not been updated by Jay for HM case
   :param trig_file_h1: Trigger file with H1 triggers
   :param trig_file_l1: Trigger file with L1 triggers
   :param trigger:
       Row of processedclist with strong trigger, only used to fix the time
   :param time_lf: Linear-free time, used if trigger wasn't given
   :param det_ind: Index of strong detector (0 for H1, 1 for L1)
   :param max_time_slide_shift: Max delay allowed for background triggers
   :param time_shift_tol: Maximum physical delay for zero-lag triggers
   :param score_reduction_max:
       Allow a reduction in SNR^2 of max * params.MAX_FRIEND_DEGRADE_SNR2 +
       score_reduction_max when keeping friends
   :param minimal_time_slide_jump: Jumps in timeslides
   :param score_func:
       Function to take two triggers and return the coherent score
   :param verbose: Flag to print progress
   :return: Background and zero-lag events as in coincidence, without cut
