triggers_single_detector_HM.TriggerList.inject_wf_into_data
===========================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Injects waveform into strain data, read description of injection_args for time convention, modifies strain in place as strain is a mutable type

Signature
---------

.. code-block:: python

   def inject_wf_into_data(times, strain, bank, asdfunc, whitened = False, taper_wt_filter = False, taper_fraction = 0.2, min_filt_trunc_time = 1, **injection_args)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``times``
     - -
     - -
     - Array with times (s)
   * - ``strain``
     - -
     - -
     - Array with strain data
   * - ``bank``
     - -
     - -
     - Instance of TemplateBank
   * - ``asdfunc``
     - -
     - -
     - Function returning ASDs (Hz^-0.5) given frequencies (Hz)
   * - ``whitened``
     - -
     - False
     - Flag indicating whether the strain data is raw or whitened
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
   * - ``\*\*injection_args``
     - -
     - -
     - Parameters to inject waveform(s) into strain data time = Scalar or list with absolute GPS time(s) (taken to be right edges for pars/wf, linear-free times for calphas, fft overlap convention (right edge + 1)) snr = Scalar or list with SNR(s) phase = Scalar or list with phase(s) type = Either 'par', 'pars', 'calpha', 'calphas', 'wf', or 'wfs' inj_pars = 1D or 2D arrays with m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2, or calphas, or time-domain waveforms approximant = Approximant

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Inserts waveform into strain data, returns noiseless stream with injection(s)

Docstring
---------

.. code-block:: text

   Injects waveform into strain data, read description of injection_args
   for time convention, modifies strain in place as strain is a mutable type
   :param times: Array with times (s)
   :param strain: Array with strain data
   :param bank: Instance of TemplateBank
   :param asdfunc: Function returning ASDs (Hz^-0.5) given frequencies (Hz)
   :param whitened:
       Flag indicating whether the strain data is raw or whitened
   :param taper_wt_filter:
       Flag whether to taper the time domain response of the whitening filter
       with a Tukey window
   :param taper_fraction:
       Fraction of response to taper with a Tukey window, if applicable
       (0 is boxcar, 1 is Hann)
   :param injection_args:
       Parameters to inject waveform(s) into strain data
       time = Scalar or list with absolute GPS time(s) (taken to be right
              edges for pars/wf, linear-free times for calphas,
              fft overlap convention (right edge + 1))
       snr =  Scalar or list with SNR(s)
       phase = Scalar or list with phase(s)
       type = Either 'par', 'pars', 'calpha', 'calphas', 'wf', or 'wfs'
       inj_pars = 1D or 2D arrays with
           m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2,
           or calphas,
           or time-domain waveforms
       approximant = Approximant
   :return:
       Inserts waveform into strain data, returns noiseless stream with
       injection(s)
