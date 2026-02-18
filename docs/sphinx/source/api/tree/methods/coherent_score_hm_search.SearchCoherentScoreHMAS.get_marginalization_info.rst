coherent_score_hm_search.SearchCoherentScoreHMAS.get_marginalization_info
=========================================================================

Back to :doc:`Class page <../classes/coherent_score_hm_search.SearchCoherentScoreHMAS>`

Summary
-------

Return a MarginalizationInfoHM object with extrinsic parameter integration results, ensuring that one of three conditions regarding the effective sample size holds: \\\* n_effective >= .min_n_effective; or \\\* n_qmc == 2 \\\*\\\* .max_log2n_qmc; or \\\* n_effective = 0 (if the first proposal only gave unphysical samples)

Signature
---------

.. code-block:: python

   def get_marginalization_info(self, dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dh_mtd``
     - -
     - -
     - -
   * - ``hh_md``
     - -
     - -
     - -
   * - ``times``
     - -
     - -
     - -
   * - ``incoherent_lnprob_td``
     - -
     - -
     - -
   * - ``mode_ratios_qm``
     - -
     - -
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

   Return a MarginalizationInfoHM object with extrinsic parameter
   integration results, ensuring that one of three conditions
   regarding the effective sample size holds:
       * n_effective >= .min_n_effective; or
       * n_qmc == 2 ** .max_log2n_qmc; or
       * n_effective = 0 (if the first proposal only gave
                          unphysical samples)
