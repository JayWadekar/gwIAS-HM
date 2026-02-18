coherent_score_hm_search.SearchCoherentScoreHMAS.lnlike_marginalized
====================================================================

Back to :doc:`Class page <../classes/coherent_score_hm_search.SearchCoherentScoreHMAS>`

Summary
-------

Return log of marginalized likelihood over inclination, sky location, orbital phase, polarization, time of arrival and distance.

Signature
---------

.. code-block:: python

   def lnlike_marginalized(self, dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dh_mtd``
     - (n_modes, n_times, n_det) complex array
     - -
     - Timeseries of the inner product (d\\\|h) between data and template, where the template is evaluated at a distance \\\`\\\`self.lookup_table.REFERENCE_DISTANCE\\\`\\\` Mpc. The convention in the inner product is that the second factor (i.e. h, not d) is conjugated.
   * - ``hh_md``
     - (n_modes\\\*(n_modes-1)/2, n_det) complex array
     - -
     - Covariance between the different modes, i.e. (h_m\\\|h_m'), with the off-diagonal entries (m != m') multiplied by 2. The ordering of the modes can be found with \\\`self.m_arr[self.m_inds], self.m_arr[self.mprime_inds]\\\`. The same template normalization and inner product convention as for \\\`\\\`dh_mtd\\\`\\\` apply.
   * - ``times``
     - (n_times,) float array
     - -
     - Times corresponding to the (d\\\|h) timeseries (s).
   * - ``incoherent_lnprob_td``
     - (n_times, n_det) float array
     - -
     - Incoherent proposal for log probability of arrival times at each detector.
   * - ``mode_ratios_qm``
     - (2\\\*\\\*self.max_log2n_qmc, n_modes-1) float array
     - -
     - Samples of mode amplitude ratio to the first mode (the part independent of inclination). These samples are used to marginalize over intrinsic parameters, mainly mass ratio.

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

   Return log of marginalized likelihood over inclination, sky
   location, orbital phase, polarization, time of arrival and
   distance.
   
   Parameters
   ----------
   dh_mtd: (n_modes, n_times, n_det) complex array
       Timeseries of the inner product (d|h) between data and
       template, where the template is evaluated at a distance
       ``self.lookup_table.REFERENCE_DISTANCE`` Mpc.
       The convention in the inner product is that the second
       factor (i.e. h, not d) is conjugated.
   
   hh_md: (n_modes*(n_modes-1)/2, n_det) complex array
       Covariance between the different modes, i.e. (h_m|h_m'),
       with the off-diagonal entries (m != m') multiplied by 2.
       The ordering of the modes can be found with
       `self.m_arr[self.m_inds], self.m_arr[self.mprime_inds]`.
       The same template normalization and inner product convention
       as for ``dh_mtd`` apply.
   
   times: (n_times,) float array
       Times corresponding to the (d|h) timeseries (s).
   
   incoherent_lnprob_td: (n_times, n_det) float array
       Incoherent proposal for log probability of arrival times at
       each detector.
   
   mode_ratios_qm: (2**self.max_log2n_qmc, n_modes-1) float array
       Samples of mode amplitude ratio to the first mode (the part
       independent of inclination). These samples are used to
       marginalize over intrinsic parameters, mainly mass ratio.
