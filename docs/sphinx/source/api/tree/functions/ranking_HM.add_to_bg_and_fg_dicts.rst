ranking_HM.add_to_bg_and_fg_dicts
=================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Populates events in lists of pure background, zero-lag candidates, LSC events, and injected events

Signature
---------

.. code-block:: python

   def add_to_bg_and_fg_dicts(events, loc_id, bg_fg_dicts, extra_arrays = (), separate_lsc_events = True, separate_injections = True, inj_ends = INJECTION_ENDS, inj_starts = INJECTION_STARTS, eps = 4, output_mask_inj = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``events``
     - -
     - -
     - Array of shape n_events x (n_det=2) x row of processedclist
   * - ``loc_id``
     - -
     - -
     - Tuple with (chirp_mass_id, subbank ID)
   * - ``bg_fg_dicts``
     - -
     - -
     - List of dicts with pure background, zero-lag candidates, (and separate LSC events and injected events, if needed)
   * - ``extra_arrays``
     - -
     - ()
     - If known, iterable with extra arrays for the events (timeseries, veto_metadata, coherent scores)
   * - ``separate_lsc_events``
     - -
     - True
     - Flag to collect lsc events separately
   * - ``separate_injections``
     - -
     - True
     - Flag to collect injections separately
   * - ``inj_ends``
     - -
     - INJECTION_ENDS
     - Array with end times of injections
   * - ``inj_starts``
     - -
     - INJECTION_STARTS
     - Array with start times of injections
   * - ``eps``
     - -
     - 4
     - Tolerance for checking if an event is close to an injection
   * - ``output_mask_inj``
     - -
     - False
     - Flag to output the mask of injected events

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Adds an entry to each dict in bg_fg_dicts with key = loc_id, and item = events, or (events, timeseries) if we have timeseries

Docstring
---------

.. code-block:: text

   Populates events in lists of pure background, zero-lag candidates,
   LSC events, and injected events
   :param events: Array of shape n_events x (n_det=2) x row of processedclist
   :param loc_id: Tuple with (chirp_mass_id, subbank ID)
   :param bg_fg_dicts:
       List of dicts with pure background, zero-lag candidates,
        (and separate LSC events and injected events, if needed)
   :param extra_arrays:
       If known, iterable with extra arrays for the events
       (timeseries, veto_metadata, coherent scores)
   :param separate_lsc_events: Flag to collect lsc events separately
   :param separate_injections: Flag to collect injections separately
   :param inj_ends: Array with end times of injections
   :param inj_starts: Array with start times of injections
   :param eps: Tolerance for checking if an event is close to an injection
   :param output_mask_inj: Flag to output the mask of injected events
   :return:
       Adds an entry to each dict in bg_fg_dicts with key = loc_id, and
       item = events, or (events, timeseries) if we have timeseries
