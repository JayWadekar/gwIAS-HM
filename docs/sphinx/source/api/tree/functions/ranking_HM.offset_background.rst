ranking_HM.offset_background
============================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Finds the amount to shift the detectors' data streams by

Signature
---------

.. code-block:: python

   def offset_background(dt, time_slide_jump, dt_shift)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dt``
     - -
     - -
     - Time delays (s)
   * - ``time_slide_jump``
     - -
     - -
     - The least count of time slides (s)
   * - ``dt_shift``
     - -
     - -
     - Time resolution (s)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Amount to add to detector to bring the timeseries to zero lag wrt the reference detector (opposite convention to coherent_score_mz.py)

Docstring
---------

.. code-block:: text

   Finds the amount to shift the detectors' data streams by
   :param dt: Time delays (s)
   :param time_slide_jump: The least count of time slides (s)
   :param dt_shift: Time resolution (s)
   :return:
       Amount to add to detector to bring the timeseries to zero lag
       wrt the reference detector (opposite convention to coherent_score_mz.py)
