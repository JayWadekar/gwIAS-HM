triggers_single_detector_HM.TriggerList.gen_triggers_local
==========================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Generates triggers at/on a small calpha grid around a trigger Has some not so quantifiable losses/biases due to the truncation of the waveforms that are being compared, and interpolations of the hole and PSD drift corrections Note: If the location is beyond the length of the data, it actually generates triggers around the edge due to quirks of searchsorted

Signature
---------

.. code-block:: python

   def gen_triggers_local(self, trigger = None, location = None, dt_left = params.DT_OPT, dt_right = params.DT_OPT, subset_defined = False, avoid_ids = None, compute_calphas = None, apply_threshold = True, relative_binning = True, delta = 0.1, relative_freq_bins = None, zero_pad = True, best_only = False, orthogonalize_modes = True, return_mode_covariance = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - None
     - Trigger in the form of a row of a processed clist
   * - ``location``
     - -
     - None
     - Tuple of length 2 with (linear-free time, calphas), used if the trigger was not given
   * - ``dt_left``
     - -
     - params.DT_OPT
     - Keep triggers with (t_trig - dt_left) <= t_lf
   * - ``dt_right``
     - -
     - params.DT_OPT
     - Keep triggers with t_lf <= (t_trig + dt_right)
   * - ``subset_defined``
     - -
     - False
     - Flag indicating if we already defined the required subset of data
   * - ``avoid_ids``
     - -
     - None
     - If needed, pass list of trigger IDs to avoid computing triggers and save on computations during coincidence
   * - ``compute_calphas``
     - -
     - None
     - If needed, pass list/2D np.array of calphas to compute triggers for, overrides the default set to iterate over
   * - ``apply_threshold``
     - -
     - True
     - Flag indicating whether to apply SNR cut
   * - ``relative_binning``
     - -
     - True
     - Flag indicating whether we are using relative binning
   * - ``delta``
     - -
     - 0.1
     - Phase allowed to accumulate within each bin
   * - ``relative_freq_bins``
     - -
     - None
     - Array with bin edges for frequency interpolation
   * - ``zero_pad``
     - -
     - True
     - Flag indicating whether to zero pad, or pad with existing data
   * - ``best_only``
     - -
     - False
     - Flag whether to return only the best trigger for each calpha, useful for making heatmaps
   * - ``orthogonalize_modes``
     - -
     - True
     - Orthogonalizes the scores from different modes
   * - ``return_mode_covariance``
     - -
     - False
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
     - Processedclist with triggers within dt_allowed

Docstring
---------

.. code-block:: text

   Generates triggers at/on a small calpha grid around a trigger
   Has some not so quantifiable losses/biases due to the truncation of
   the waveforms that are being compared, and interpolations of the hole
   and PSD drift corrections
   Note: If the location is beyond the length of the data, it actually generates
   triggers around the edge due to quirks of searchsorted
   :param trigger: Trigger in the form of a row of a processed clist
   :param location:
       Tuple of length 2 with (linear-free time, calphas), used if the
       trigger was not given
   :param dt_left: Keep triggers with (t_trig - dt_left) <= t_lf
   :param dt_right: Keep triggers with t_lf <= (t_trig + dt_right)
   :param subset_defined:
       Flag indicating if we already defined the required subset of data
   :param avoid_ids:
       If needed, pass list of trigger IDs to avoid computing
       triggers and save on computations during coincidence
   :param compute_calphas:
       If needed, pass list/2D np.array of calphas to compute triggers
       for, overrides the default set to iterate over
   :param apply_threshold: Flag indicating whether to apply SNR cut
   :param relative_binning:
       Flag indicating whether we are using relative binning
   :param delta: Phase allowed to accumulate within each bin
   :param relative_freq_bins:
       Array with bin edges for frequency interpolation
   :param zero_pad:
       Flag indicating whether to zero pad, or pad with existing data
   :param best_only:
       Flag whether to return only the best trigger for each calpha,
       useful for making heatmaps
   :param orthogonalize_modes: Orthogonalizes the scores from different modes
   :return: Processedclist with triggers within dt_allowed
