triggers_single_detector_HM.TriggerList.get_time_index
======================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

"

Signature
---------

.. code-block:: python

   def get_time_index(self, triggers)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``triggers``
     - -
     - -
     - Processedclist with triggers that jumped (can be a vector for n_triggers = 1)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Indices closest to right-edge of template + 1 = score 2. Subgrid shifts to the nearest time on the grid (s) (scalars if n_triggers = 1)

Docstring
---------

.. code-block:: text

   "
   :param triggers:
       Processedclist with triggers that jumped
       (can be a vector for n_triggers = 1)
   :return: 1. Indices closest to right-edge of template + 1 = score
            2. Subgrid shifts to the nearest time on the grid (s)
            (scalars if n_triggers = 1)
