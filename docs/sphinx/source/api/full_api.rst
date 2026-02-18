Full API Directory
==================

This page is a quick module-by-module directory for the repository API.

.. list-table:: Pipeline module directory
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Purpose
   * - :doc:`generated/triggers_single_detector_HM`
     - Single-detector triggering engine and trigger lifecycle.
   * - :doc:`generated/data_operations`
     - PSD estimation, whitening, line/hole/glitch processing.
   * - :doc:`generated/coincidence_HM`
     - Multi-detector coincidence, vetoing, and candidate output.
   * - :doc:`generated/coherent_score_hm_search`
     - HM coherent score marginalization implementation.
   * - :doc:`generated/coherent_score_mz_fast`
     - Fast/legacy coherent-scoring support utilities.
   * - :doc:`generated/ranking_HM`
     - Candidate aggregation and ranking logic.
   * - :doc:`generated/template_bank_generator_HM`
     - HM template bank and subbank generation/usage.
   * - :doc:`generated/template_bank_params_O3a_HM`
     - O3a/O3b HM bank hyperparameter definitions.
   * - :doc:`generated/triggering_on_cluster`
     - Cluster submission and run-management helpers.
   * - :doc:`generated/gw_detect_file`
     - CLI entrypoint for per-file trigger generation.
   * - :doc:`generated/utils`
     - Shared utility and path/run helpers.
   * - :doc:`generated/python_utils`
     - Low-level Python/numeric helper functions.
   * - :doc:`generated/readligo`
     - LIGO frame/HDF5 loading and segment utilities.
   * - :doc:`generated/download_data`
     - GWOSC download helper utilities.
   * - :doc:`generated/ML_modules`
     - Optional ML modules for priors/posteriors.
   * - :doc:`generated/params`
     - Global pipeline constants and thresholds.
