utils.condition_filter
======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Compute support, weight, and truncate input frequency domain filter

Signature
---------

.. code-block:: python

   def condition_filter(filt, support = None, truncate = False, shorten = False, taper = False, notch_pars = None, flen = None, wfac = params.WFAC_FILT, in_domain = 'fd', out_domain = 'fd', def_fftsize = params.DEF_FFTSIZE, taper_fraction = 0.2, min_trunc_len = 2)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``filt``
     - -
     - -
     - If in_domain is 'fd': Complex array of length nrfft(len(data)) with RFFT of {whitening filter with time domain weight at the edges} else: Real array of length len(data) with whitening filter with time domain weight at the edges
   * - ``support``
     - -
     - None
     - Support to truncate at (in indices). If None, it is calculated from the filter itself
   * - ``truncate``
     - -
     - False
     - Flag whether to truncate the time-domain support of the filter
   * - ``shorten``
     - -
     - False
     - Flag whether to shorten the time-domain length of the filter
   * - ``taper``
     - -
     - False
     - Flag whether to taper the time domain response of the filter with a Tukey window. Applied after truncating if truncate==True, else on the entire length
   * - ``notch_pars``
     - -
     - None
     - Optional parameters of notches to apply, output of notch_filter_sos
   * - ``flen``
     - -
     - None
     - Total length of filter in time domain (zero-padded in frequency domain if needed)
   * - ``wfac``
     - -
     - params.WFAC_FILT
     - (1 - Weight of the filter to capture)/2
   * - ``in_domain``
     - -
     - 'fd'
     - Domain of input ('fd' or 'td')
   * - ``out_domain``
     - -
     - 'fd'
     - Domain of output ('fd' or 'td')
   * - ``def_fftsize``
     - -
     - params.DEF_FFTSIZE
     - Default fftsize, one of the limits for the time domain length of the output
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
     - 1. If out_domain is 'fd': Complex array of the same size as filt with RFFT of conditioned filter else: Real array of the same size as filt with conditioned filter 2. Support of filter (TD filter has 2 \\\* support - 1 nonzero coeffs) 3. Total weight of filter (sum w(t)^2)

Docstring
---------

.. code-block:: text

   Compute support, weight, and truncate input frequency domain filter
   :param filt:
       If in_domain is 'fd':
           Complex array of length nrfft(len(data)) with RFFT of
           {whitening filter with time domain weight at the edges}
       else:
           Real array of length len(data) with whitening filter with time
           domain weight at the edges
   :param support:
       Support to truncate at (in indices). If None, it is calculated from the
       filter itself
   :param truncate:
       Flag whether to truncate the time-domain support of the filter
   :param shorten:
       Flag whether to shorten the time-domain length of the filter
   :param taper:
       Flag whether to taper the time domain response of the filter with a
       Tukey window. Applied after truncating if truncate==True, else on the
       entire length
   :param notch_pars:
       Optional parameters of notches to apply, output of notch_filter_sos
   :param flen:
       Total length of filter in time domain (zero-padded in frequency domain
       if needed)
   :param wfac: (1 - Weight of the filter to capture)/2
   :param in_domain: Domain of input ('fd' or 'td')
   :param out_domain: Domain of output ('fd' or 'td')
   :param def_fftsize:
       Default fftsize, one of the limits for the time domain length of the
       output
   :param taper_fraction:
       Fraction of response to taper with a Tukey window, if applicable
       (0 is boxcar, 1 is Hann)
   :param min_trunc_len: minimum N_samples//2 after truncation
   :return: 1. If out_domain is 'fd':
                   Complex array of the same size as filt with RFFT of
                   conditioned filter
               else:
                   Real array of the same size as filt with conditioned filter
            2. Support of filter (TD filter has 2 * support - 1 nonzero coeffs)
            3. Total weight of filter (sum w(t)^2)
