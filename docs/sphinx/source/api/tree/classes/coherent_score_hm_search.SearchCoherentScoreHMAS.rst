coherent_score_hm_search.SearchCoherentScoreHMAS
================================================

Back to :doc:`Module page <../modules/coherent_score_hm_search>`

Summary
-------

Class to marginalize the likelihood over extrinsic parameters, intended for the search pipeline.

.. list-table:: Methods
   :header-rows: 1

   * - Method
     - Summary
   * - :doc:`get_marginalization_info <../methods/coherent_score_hm_search.SearchCoherentScoreHMAS.get_marginalization_info>`
     - Return a MarginalizationInfoHM object with extrinsic parameter integration results, ensuring that one of three conditions regarding the effective sample size holds: \\\* n_effective >= .min_n_effective; or \\\* n_qmc == 2 \\\*\\\* .max_log2n_qmc; or \\\* n_effective = 0 (if the first proposal only gave unphysical samples)
   * - :doc:`lnlike_marginalized <../methods/coherent_score_hm_search.SearchCoherentScoreHMAS.lnlike_marginalized>`
     - Return log of marginalized likelihood over inclination, sky location, orbital phase, polarization, time of arrival and distance.

Class docstring
---------------

.. code-block:: text

   Class to marginalize the likelihood over extrinsic parameters,
   intended for the search pipeline.
   
   Assumptions:
       * Quasicircular
       * Aligned spins,
       * (l, m) = [(2, 2), (3, 3), (4, 4)] harmonics.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../methods/coherent_score_hm_search.SearchCoherentScoreHMAS.get_marginalization_info
   ../methods/coherent_score_hm_search.SearchCoherentScoreHMAS.lnlike_marginalized
