API Tree
========

This API is organized as a navigation tree:

1. Module pages
2. Per-module tables for top-level functions and classes
3. Per-class tables for methods
4. Per-callable pages with signature, input variables, and output variables

.. list-table:: Modules
   :header-rows: 1

   * - Module
     - Purpose
   * - :doc:`ML_modules <modules/ML_modules>`
     - Optional ML prior/posterior helpers used by ranking/scoring stages.
   * - :doc:`coherent_score_hm_search <modules/coherent_score_hm_search>`
     - Coherent score marginalization routines for HM searches.
   * - :doc:`coherent_score_mz_fast <modules/coherent_score_mz_fast>`
     - Fast and legacy coherent-scoring support utilities.
   * - :doc:`coincidence_HM <modules/coincidence_HM>`
     - Multi-detector coincidence, vetoing, and candidate output logic.
   * - :doc:`data_operations <modules/data_operations>`
     - PSD estimation, whitening, and glitch/line mitigation utilities.
   * - :doc:`download_data <modules/download_data>`
     - GWOSC data download helpers.
   * - :doc:`gw_detect_file <modules/gw_detect_file>`
     - CLI entry points for per-file single-detector trigger generation.
   * - :doc:`params <modules/params>`
     - Global constants and threshold parameters used across the pipeline.
   * - :doc:`python_utils <modules/python_utils>`
     - Numeric and Python utility helpers shared across modules.
   * - :doc:`ranking_HM <modules/ranking_HM>`
     - Candidate aggregation, reweighting, and ranking logic.
   * - :doc:`readligo <modules/readligo>`
     - LIGO frame/HDF5 loading and segment-manipulation helpers.
   * - :doc:`template_bank_generator_HM <modules/template_bank_generator_HM>`
     - HM template-bank generation and basis-coefficient utilities.
   * - :doc:`template_bank_params_O3a_HM <modules/template_bank_params_O3a_HM>`
     - Template-bank hyperparameter settings for observing runs.
   * - :doc:`triggering_on_cluster <modules/triggering_on_cluster>`
     - Cluster submission and trigger-run orchestration helpers.
   * - :doc:`triggers_single_detector_HM <modules/triggers_single_detector_HM>`
     - Single-detector triggering engine and trigger lifecycle.
   * - :doc:`utils <modules/utils>`
     - Cross-cutting utility functions and run/path helpers.

.. toctree::
   :hidden:
   :maxdepth: 1

   modules/ML_modules
   modules/coherent_score_hm_search
   modules/coherent_score_mz_fast
   modules/coincidence_HM
   modules/data_operations
   modules/download_data
   modules/gw_detect_file
   modules/params
   modules/python_utils
   modules/ranking_HM
   modules/readligo
   modules/template_bank_generator_HM
   modules/template_bank_params_O3a_HM
   modules/triggering_on_cluster
   modules/triggers_single_detector_HM
   modules/utils
