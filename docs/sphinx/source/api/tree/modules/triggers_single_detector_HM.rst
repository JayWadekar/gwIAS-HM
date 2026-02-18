triggers_single_detector_HM
===========================

Back to :doc:`API tree index <../index>`

Purpose
-------

Single-detector triggering engine and trigger lifecycle.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`ensure_abspath <../functions/triggers_single_detector_HM.ensure_abspath>`
     - Ensures that the path to fname is absolute, assumes it is saved in the same directory as config_fname!

.. list-table:: Classes
   :header-rows: 1

   * - Class
     - Summary
   * - :doc:`TriggerList <../classes/triggers_single_detector_HM.TriggerList>`
     - Class that takes in directories with data and template bank information, and generates/analyzes triggers # TODO: Remove notch_wt_filter options etc

.. toctree::
   :hidden:
   :maxdepth: 1

   ../classes/triggers_single_detector_HM.TriggerList
   ../functions/triggers_single_detector_HM.ensure_abspath
