coincidence_HM.secondary_peak_reject
====================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Return True if this trigger should be rejected for having a secondary peak that ruins the coherent score

Signature
---------

.. code-block:: python

   def secondary_peak_reject(processed_clist, timeseries, max_friend_degrade_snr2 = params.MAX_FRIEND_DEGRADE_SNR2, score_reduction_max = 5)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``processed_clist``
     - -
     - -
     - -
   * - ``timeseries``
     - -
     - -
     - -
   * - ``max_friend_degrade_snr2``
     - -
     - params.MAX_FRIEND_DEGRADE_SNR2
     - -
   * - ``score_reduction_max``
     - -
     - 5
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
     - -

Docstring
---------

.. code-block:: text

   Return True if this trigger should be rejected for having a secondary peak that
       ruins the coherent score
