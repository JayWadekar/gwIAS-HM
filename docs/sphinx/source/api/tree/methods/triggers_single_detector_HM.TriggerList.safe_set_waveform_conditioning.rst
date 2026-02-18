triggers_single_detector_HM.TriggerList.safe_set_waveform_conditioning
======================================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Sets waveform conditioning given the ASD, and expands fftsize by a factor of two if needed to make it work. Useful when trialing small FFTsizes for BNS-like banks

Signature
---------

.. code-block:: python

   def safe_set_waveform_conditioning(bank, fftsize, dt, asdfunc = None, shorten_fftsize = False, taper_wt_filter = False, taper_fraction = 0.2, min_filt_trunc_time = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``bank``
     - -
     - -
     - Instance of TemplateBank
   * - ``fftsize``
     - -
     - -
     - Integer with input candidate fftsize
   * - ``dt``
     - -
     - -
     - Sampling interval of template (in seconds)
   * - ``asdfunc``
     - -
     - None
     - Function returning ASDs (Hz^-0.5) given frequencies (Hz)
   * - ``shorten_fftsize``
     - -
     - False
     - If True, we try to condition with a shorter fftsize if possible
   * - ``taper_wt_filter``
     - -
     - False
     - Flag whether to taper the time domain response of the whitening filter with a Tukey window
   * - ``taper_fraction``
     - -
     - 0.2
     - Fraction of response to taper with a Tukey window, if applicable (0 is boxcar, 1 is Hann)
   * - ``min_filt_trunc_time``
     - -
     - 1
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
     - 1. FFTsize for which conditioning worked (either input or 2x input) 2. Whitening filter in the Fourier domain 3. Half-support of the whitening filter in the time-domain

Docstring
---------

.. code-block:: text

   Sets waveform conditioning given the ASD, and expands fftsize by a factor
   of two if needed to make it work. Useful when trialing small FFTsizes for
   BNS-like banks
   :param bank: Instance of TemplateBank
   :param fftsize: Integer with input candidate fftsize
   :param dt: Sampling interval of template (in seconds)
   :param asdfunc: Function returning ASDs (Hz^-0.5) given frequencies (Hz)
   :param shorten_fftsize:
       If True, we try to condition with a shorter fftsize if possible
   :param taper_wt_filter:
       Flag whether to taper the time domain response of the whitening filter
       with a Tukey window
   :param taper_fraction:
       Fraction of response to taper with a Tukey window, if applicable
       (0 is boxcar, 1 is Hann)
   :return:
       1. FFTsize for which conditioning worked (either input or 2x input)
       2. Whitening filter in the Fourier domain
       3. Half-support of the whitening filter in the time-domain
