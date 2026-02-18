utils.condition_filter_td
=========================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Compute support, weight, and truncate input time domain filter

Signature
---------

.. code-block:: python

   def condition_filter_td(filt_td, support = None, truncate = False, taper = False, wfac = params.WFAC_FILT, taper_fraction = 0.2, min_trunc_len = 2)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``filt_td``
     - -
     - -
     - Array with time domain filter with weight at the edges
   * - ``support``
     - -
     - None
     - Support to truncate at (in indices). If None, it is calculated from the filter itself
   * - ``truncate``
     - -
     - False
     - Flag indicating whether to truncate time-domain filter
   * - ``taper``
     - -
     - False
     - Flag whether to taper the time domain response of the filter with a Tukey window. Applied ater truncating if truncate==True, else on the entire length
   * - ``wfac``
     - -
     - params.WFAC_FILT
     - (1 - Weight of the filter to capture)/2
   * - ``taper_fraction``
     - -
     - 0.2
     - Fraction of response to taper with a Tukey window, if applicable (0 is boxcar, 1 is Hann)
   * - ``min_trunc_len``
     - -
     - 2
     - minimum N_samples//2 after truncation

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Array of the same size as filt_td with conditioned filter 2. Support of filter (TD filter has 2 \\\* support - 1 nonzero coeffs) 3. Total weight of filter (sum w(t)^2)

Docstring
---------

.. code-block:: text

   Compute support, weight, and truncate input time domain filter
   :param filt_td: Array with time domain filter with weight at the edges
   :param support: Support to truncate at (in indices). If None, it is
                   calculated from the filter itself
   :param truncate: Flag indicating whether to truncate time-domain filter
   :param taper:
       Flag whether to taper the time domain response of the filter with a
       Tukey window. Applied ater truncating if truncate==True, else on the
       entire length
   :param wfac: (1 - Weight of the filter to capture)/2
   :param taper_fraction:
       Fraction of response to taper with a Tukey window, if applicable
       (0 is boxcar, 1 is Hann)
   :param min_trunc_len: minimum N_samples//2 after truncation
   :return: 1. Array of the same size as filt_td with conditioned filter
            2. Support of filter (TD filter has 2 * support - 1 nonzero coeffs)
            3. Total weight of filter (sum w(t)^2)
