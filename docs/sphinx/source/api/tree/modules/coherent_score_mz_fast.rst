coherent_score_mz_fast
======================

Back to :doc:`API tree index <../index>`

Purpose
-------

Fast and legacy coherent-scoring support utilities.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`coherent_score_montecarlo_sky <../functions/coherent_score_mz_fast.coherent_score_montecarlo_sky>`
     - # TODO: Avoid tuples, make varargs? Evaluates the coherent score integral by montecarlo sampling all relevant variables
   * - :doc:`create_samples <../functions/coherent_score_mz_fast.create_samples>`
     - Create samples and save to file
   * - :doc:`create_time_dict <../functions/coherent_score_mz_fast.create_time_dict>`
     - Creates dictionary indexed by time, giving RA-Dec pairs for montecarlo integration
   * - :doc:`delays <../functions/coherent_score_mz_fast.delays>`
     - Computes delays in arrival times between detectors, referred to the first (beyond 2 delays for most, and 3 delays for all, they're overspecified, but that's OK) Always a 1D array
   * - :doc:`dt2key <../functions/coherent_score_mz_fast.dt2key>`
     - Jitable key for the dictionary from the time delays This is clunky but faster than the nice expression by O(1) when vectorized
   * - :doc:`gen_sample_amps_from_fplus_fcross <../functions/coherent_score_mz_fast.gen_sample_amps_from_fplus_fcross>`
     - -
   * - :doc:`lg_approx <../functions/coherent_score_mz_fast.lg_approx>`
     - No docstring summary available.
   * - :doc:`lg_fast <../functions/coherent_score_mz_fast.lg_fast>`
     - Fast evaluation of the marginalized piece via an interpolation table for a vector x, much faster via vectors than many scalars due to vectorized interp
   * - :doc:`lgg <../functions/coherent_score_mz_fast.lgg>`
     - param: zthatthatz
   * - :doc:`marg_lk <../functions/coherent_score_mz_fast.marg_lk>`
     - Computes likelihood marginalized over distance and orbital phase
   * - :doc:`phase_lags <../functions/coherent_score_mz_fast.phase_lags>`
     - Computes phase lags between detectors, referred to the first (beyond 2 delays for most, and 3 delays for all, they're overspecified, but that's OK)
   * - :doc:`rand_choice_nb <../functions/coherent_score_mz_fast.rand_choice_nb>`
     - -

.. list-table:: Classes
   :header-rows: 1

   * - Class
     - Summary
   * - :doc:`CoherentScoreMZ <../classes/coherent_score_mz_fast.CoherentScoreMZ>`
     - No class docstring summary available.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../classes/coherent_score_mz_fast.CoherentScoreMZ
   ../functions/coherent_score_mz_fast.coherent_score_montecarlo_sky
   ../functions/coherent_score_mz_fast.create_samples
   ../functions/coherent_score_mz_fast.create_time_dict
   ../functions/coherent_score_mz_fast.delays
   ../functions/coherent_score_mz_fast.dt2key
   ../functions/coherent_score_mz_fast.gen_sample_amps_from_fplus_fcross
   ../functions/coherent_score_mz_fast.lg_approx
   ../functions/coherent_score_mz_fast.lg_fast
   ../functions/coherent_score_mz_fast.lgg
   ../functions/coherent_score_mz_fast.marg_lk
   ../functions/coherent_score_mz_fast.phase_lags
   ../functions/coherent_score_mz_fast.rand_choice_nb
