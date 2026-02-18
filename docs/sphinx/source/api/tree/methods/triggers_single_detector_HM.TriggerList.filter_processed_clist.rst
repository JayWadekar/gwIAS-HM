triggers_single_detector_HM.TriggerList.filter_processed_clist
==============================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Filters a processed clist

Signature
---------

.. code-block:: python

   def filter_processed_clist(processedclist, filters = None, rejects = None, t0 = 0.0, c0_pos = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``processedclist``
     - -
     - -
     - Must be a numpy array, not a list
   * - ``filters``
     - -
     - None
     - See filter_triggers for values
   * - ``rejects``
     - -
     - None
     - -
   * - ``t0``
     - -
     - 0.0
     - -
   * - ``c0_pos``
     - -
     - None
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
     - filteredclist

Docstring
---------

.. code-block:: text

   Filters a processed clist
   :param processedclist: Must be a numpy array, not a list
   :param filters: See filter_triggers for values
   :param rejects:
   :param t0:
   :param c0_pos:
   :return: filteredclist
