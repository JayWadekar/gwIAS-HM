coincidence_HM.veto_and_optimize_single_detector
================================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Veto and optimize coincident candidates in a single detector

Signature
---------

.. code-block:: python

   def veto_and_optimize_single_detector(triggers, trig_obj, time_shift_tol, group_duration = 0.1, veto_triggers = True, min_veto_chi2 = None, apply_threshold = True, origin = 0, n_cores = 1, opt_format = 'new')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``triggers``
     - -
     - -
     - Processedclist with candidates in a detector
   * - ``trig_obj``
     - -
     - -
     - Instance of trig.TriggerList with processed data in the detector
   * - ``time_shift_tol``
     - -
     - -
     - Veto only one trigger every bucket of time_shift_tol seconds
   * - ``group_duration``
     - -
     - 0.1
     - Divide triggers into groups every group_duration seconds and define subset for this group at once, saves on FFTs
   * - ``veto_triggers``
     - -
     - True
     - Flag indicating whether to veto triggers
   * - ``min_veto_chi2``
     - -
     - None
     - If given, veto only the candidates above this bar
   * - ``apply_threshold``
     - -
     - True
     - Flag to apply threshold on single-detector chi2 when optimizing
   * - ``origin``
     - -
     - 0
     - Origin to split the trigger times relative to
   * - ``n_cores``
     - -
     - 1
     - Number of cores to use for splitting the computation
   * - ``opt_format``
     - -
     - 'new'
     - How we choose the finer grid, changed between O1 and O2 analyses Exposed here to replicate old runs if needed

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Dictionary indexed by bucket_ids with passed/failed veto, a processedclist of friends, a snr^2 correction factor, and metadata about data quality and glitch tests (see veto_and_optimize_group for an explanation)

Docstring
---------

.. code-block:: text

   Veto and optimize coincident candidates in a single detector
   :param triggers: Processedclist with candidates in a detector
   :param trig_obj:
       Instance of trig.TriggerList with processed data in the detector
   :param time_shift_tol:
       Veto only one trigger every bucket of time_shift_tol seconds
   :param group_duration:
       Divide triggers into groups every group_duration seconds and
       define subset for this group at once, saves on FFTs
   :param veto_triggers: Flag indicating whether to veto triggers
   :param min_veto_chi2: If given, veto only the candidates above this bar
   :param apply_threshold:
       Flag to apply threshold on single-detector chi2 when optimizing
   :param origin: Origin to split the trigger times relative to
   :param n_cores: Number of cores to use for splitting the computation
   :param opt_format:
       How we choose the finer grid, changed between O1 and O2 analyses
       Exposed here to replicate old runs if needed
   :return:
       Dictionary indexed by bucket_ids with passed/failed veto,
       a processedclist of friends, a snr^2 correction factor, and metadata
       about data quality and glitch tests
       (see veto_and_optimize_group for an explanation)
