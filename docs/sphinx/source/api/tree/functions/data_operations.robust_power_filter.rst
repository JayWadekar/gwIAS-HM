data_operations.robust_power_filter
===================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def robust_power_filter(times, power_sequence, mask_times, avg_len, avoid_len = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``times``
     - -
     - -
     - Array with central times of intervals in which power was measured
   * - ``power_sequence``
     - -
     - -
     - Array with measured power in intervals
   * - ``mask_times``
     - -
     - -
     - Boolean mask indicating whether interval sees an existing hole
   * - ``avg_len``
     - -
     - -
     - Number of intervals that moving average test spans
   * - ``avoid_len``
     - -
     - 1
     - Excess power is correlated, avoid these many intervals around central one

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Array with central times, excess power with moving average subtracted, and Boolean mask into times

Docstring
---------

.. code-block:: text

   :param times: Array with central times of intervals in which power was
                 measured
   :param power_sequence: Array with measured power in intervals
   :param mask_times: Boolean mask indicating whether interval sees an
                      existing hole
   :param avg_len: Number of intervals that moving average test spans
   :param avoid_len: Excess power is correlated, avoid these many intervals
                     around central one
   :return: Array with central times, excess power with moving average
            subtracted, and Boolean mask into times
