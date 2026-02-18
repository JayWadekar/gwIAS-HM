triggers_single_detector_HM.TriggerList.prepare_subset_for_vetoes
=================================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Prepare subset that is large enough for vetos/optimization of any trigger in triggers

Signature
---------

.. code-block:: python

   def prepare_subset_for_vetoes(self, triggers = None, locations = None, dt_opt = params.DT_OPT, support = params.SUPPORT_SINC_FILTER_OPT, zero_pad = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``triggers``
     - -
     - None
     - n_trigger x len(processedclist[0]) array with triggers (can be a vector for n_trigger = 1)
   * - ``locations``
     - -
     - None
     - List of length n_locations with 2-tuples with locations (can be a single location for n_locations = 1)
   * - ``dt_opt``
     - -
     - params.DT_OPT
     - Length of buffer (s) to allow in calpha optimization
   * - ``support``
     - -
     - params.SUPPORT_SINC_FILTER_OPT
     - Support of sinc-interpolating filter
   * - ``zero_pad``
     - -
     - True
     - Flag indicating whether to zero pad, or pad with existing data

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Reference index for score in full data 2. Number of left scores guaranteed, not inclusive of ref 3. Number of right scores guaranteed, not inclusive of ref TODO: Fix support to be in units of seconds?

Docstring
---------

.. code-block:: text

   Prepare subset that is large enough for vetos/optimization of any
   trigger in triggers
   :param triggers:
       n_trigger x len(processedclist[0]) array with triggers
       (can be a vector for n_trigger = 1)
   :param locations:
       List of length n_locations with 2-tuples with locations
       (can be a single location for n_locations = 1)
   :param dt_opt: Length of buffer (s) to allow in calpha optimization
   :param support: Support of sinc-interpolating filter
   :param zero_pad:
       Flag indicating whether to zero pad, or pad with existing data
   :return: 1. Reference index for score in full data
            2. Number of left scores guaranteed, not inclusive of ref
            3. Number of right scores guaranteed, not inclusive of ref
   TODO: Fix support to be in units of seconds?
