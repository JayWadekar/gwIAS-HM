ranking_HM.generate_downsampling_correction_fits
================================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

We want to calculate the non-Gaussian correction to the ranking statistic (see arXiv: 2405.17400). We therefore want to compare the empirical histogram of the SNR^2 to that expected from the Gaussian noise hypothesis (see Fig.3 of the paper). The empirical histogram is unfortunately also affected by downsampling at the low SNR^2 end due to using HM marginalized statistic based threshold in the matched_filtering stage (instead of using rho_incoherent^2 as the threshold). To remedy this, we do a quick and rough calculation of the downsampling correction. and fit it by a parabolic function to avoid noise at the high SNR^2 end. The fit is then used to correct the empirical histogram or rank function.

Signature
---------

.. code-block:: python

   def generate_downsampling_correction_fits(Z_gauss_complex = None, bank_id = None, tbp_dir = None, bank_obj = None, n_SNRsq_bins = 15, SNRsq_cutoff = 20, return_plot = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``Z_gauss_complex``
     - -
     - None
     - Complex Gaussian noise triggers [n_triggers,3] They take time to generate, so try generate in a notebook once and supply them for all banks
   * - ``bank_id``
     - -
     - None
     - The bank id of the bank to be used for the correction (if bank_obj is not provided)
   * - ``tbp_dir``
     - -
     - None
     - The directory where the template banks are stored (you could load temp bank params obj and use tbp.DIR)
   * - ``bank_obj``
     - -
     - None
     - The bank object to be used for the correction
   * - ``n_SNRsq_bins``
     - -
     - 15
     - Number of bins to use for the histogram (the histogram can be noisy if the number of bins is too large)
   * - ``SNRsq_cutoff``
     - -
     - 20
     - The cutoff SNR^2 value above which the triggers are considered. This parameter needs further investigation, Either provide the lowest SNR^2 cutoff among all banks used during triggering (recommended), or provide a median value across all banks.
   * - ``return_plot``
     - -
     - False
     - Whether to return the plot (for debugging)

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

   We want to calculate the non-Gaussian correction
   to the ranking statistic (see arXiv: 2405.17400).
   We therefore want to compare the empirical histogram of the SNR^2
   to that expected from the Gaussian noise hypothesis (see Fig.3 of the paper).
   The empirical histogram is unfortunately also affected by downsampling
   at the low SNR^2 end due to using HM marginalized statistic based threshold
   in the matched_filtering stage (instead of using rho_incoherent^2 as the threshold).
   To remedy this, we do a quick and rough calculation of the downsampling correction.
   and fit it by a parabolic function to avoid noise at the high SNR^2 end.
   The fit is then used to correct the empirical histogram or rank function.
   :param Z_gauss_complex: Complex Gaussian noise triggers [n_triggers,3]
       They take time to generate, so try generate in a notebook once and
       supply them for all banks
   :param bank_id: The bank id of the bank to be used for the correction
                   (if bank_obj is not provided)
   :param tbp_dir: The directory where the template banks are stored
                   (you could load temp bank params obj and use tbp.DIR)
   :param bank_obj: The bank object to be used for the correction
   :param n_SNRsq_bins: Number of bins to use for the histogram
                       (the histogram can be noisy if the number of bins is too large)
   :param SNRsq_cutoff: The cutoff SNR^2 value above which the triggers are considered.
                       This parameter needs further investigation,
                       Either provide the lowest SNR^2 cutoff among all banks used
                       during triggering (recommended),
                       or provide a median value across all banks.
   :param return_plot: Whether to return the plot (for debugging)
