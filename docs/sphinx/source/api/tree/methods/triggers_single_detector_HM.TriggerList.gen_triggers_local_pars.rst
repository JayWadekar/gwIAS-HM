triggers_single_detector_HM.TriggerList.gen_triggers_local_pars
===============================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

NOTE: This fn has not been modified by Jay for higher modes. Generates triggers at/on a small calpha grid around a trigger Has some not so quantifiable losses/biases due to the truncation of the waveforms that are being compared, and interpolations of the hole and PSD drift corrections

Signature
---------

.. code-block:: python

   def gen_triggers_local_pars(self, trigger = None, location = None, dt_left = params.DT_OPT, dt_right = params.DT_OPT, subset_defined = False, relative_binning = True, delta = 0.1, relative_freq_bins = None, psd_drift_kwargs = None, **pars)

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
   * - ``psd_drift_kwargs``
     - -
     - None
     - Dictionary with any extra arguments for recomputing PSD drift
   * - ``\*\*pars``
     - -
     - -
     - Dictionary with list of parameters for generating waveforms (look at gen_wf_fd_from_pars)

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

   NOTE: This fn has not been modified by Jay for higher modes.
   Generates triggers at/on a small calpha grid around a trigger
   Has some not so quantifiable losses/biases due to the truncation of
   the waveforms that are being compared, and interpolations of the hole
   and PSD drift corrections
   :param trigger: Trigger in the form of a row of a processed clist
   :param location:
       Tuple of length 2 with (linear-free time, calphas), used if the
       trigger was not given
   :param dt_left: Keep triggers with (t_trig - dt_left) <= t_lf
   :param dt_right: Keep triggers with t_lf <= (t_trig + dt_right)
   :param subset_defined:
       Flag indicating if we already defined the required subset of data
   :param relative_binning:
       Flag indicating whether we are using relative binning
   :param delta: Phase allowed to accumulate within each bin
   :param relative_freq_bins:
       Array with bin edges for frequency interpolation
   :param psd_drift_kwargs:
       Dictionary with any extra arguments for recomputing PSD drift
   :param pars:
       Dictionary with list of parameters for generating waveforms
       (look at gen_wf_fd_from_pars)
   :return: Processedclist with triggers within dt_allowed
