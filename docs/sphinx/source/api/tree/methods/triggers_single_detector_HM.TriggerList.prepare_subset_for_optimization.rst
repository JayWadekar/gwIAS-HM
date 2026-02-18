triggers_single_detector_HM.TriggerList.prepare_subset_for_optimization
=======================================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Defines subset if needed, and returns parameters to locate trigger in it

Signature
---------

.. code-block:: python

   def prepare_subset_for_optimization(self, trigger = None, location = None, dt = params.DT_OPT, subset_defined = False, zero_pad = True)

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
     - Tuple of length 2 with (linear-free time, calphas), used if trigger is not given
   * - ``dt``
     - -
     - params.DT_OPT
     - Shift in time (s) to allow
   * - ``subset_defined``
     - -
     - False
     - Flag indicating if we already defined the required subset of data, used to save on FFTs
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
     - 1. calphas of trigger 2. Index of trigger score in global time 3. Index of trigger into subset

Docstring
---------

.. code-block:: text

   Defines subset if needed, and returns parameters to locate trigger in it
   :param trigger: Trigger in the form of a row of a processed clist
   :param location:
       Tuple of length 2 with (linear-free time, calphas), used if trigger
       is not given
   :param dt: Shift in time (s) to allow
   :param subset_defined:
       Flag indicating if we already defined the required subset of data,
       used to save on FFTs
   :param zero_pad:
       Flag indicating whether to zero pad, or pad with existing data
   :return:
       1. calphas of trigger
       2. Index of trigger score in global time
       3. Index of trigger into subset
