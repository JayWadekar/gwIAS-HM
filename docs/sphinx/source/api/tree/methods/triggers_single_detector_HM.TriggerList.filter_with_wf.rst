triggers_single_detector_HM.TriggerList.filter_with_wf
======================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Generate triggers with sinc-interpolated scores with a single template for entire data or a subset of it

Signature
---------

.. code-block:: python

   def filter_with_wf(self, wf_whitened_fd = None, calphas = None, subset = False, apply_threshold = True, zero_invalid_before_sinc_interp = True, support = params.SUPPORT_SINC_FILTER, ensure_sinc_support = False, recompute_psd_drift_correction = False, marginalized_score_HM = True, interpolate_psd_drift_correction = False, return_format = 'clist', adjust_times = 1, **psd_drift_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wf_whitened_fd``
     - -
     - None
     - Array of 3 x len(rfftfreq(fftsize)) with FFT of waveform
   * - ``calphas``
     - -
     - None
     - Optional array of calphas representing waveform. If wf_whitened_fd is None, we use the calphas to generate a waveform
   * - ``subset``
     - -
     - False
     - Flag indicating whether to filter the waveform with the entire data or subset of it
   * - ``apply_threshold``
     - -
     - True
     - Flag to apply threshold on chi^2 when collecting triggers
   * - ``zero_invalid_before_sinc_interp``
     - -
     - True
     - Flag indicating whether to zero invalid scores before sinc-interpolating. Default set to preserve search, exposed for debugging purposes
   * - ``support``
     - -
     - params.SUPPORT_SINC_FILTER
     - Support of sinc-interpolating filter
   * - ``ensure_sinc_support``
     - -
     - False
     - When sinc-interpolating inside an interval, ensure that we only retain scores within that interval. Set to False when collecting triggers to preserve older runs, use True when optimizing
   * - ``recompute_psd_drift_correction``
     - -
     - False
     - Flag indicating whether to redefine PSD drift correction, exposed for debugging with astrophysical waveforms
   * - ``marginalized_score_HM``
     - -
     - True
     - Use a semi-marginalized score instead of \\\|Z22\\\|\\\*\\\*2+\\\|Z33\\\|\\\*\\\*2+\\\|Z44\\\|\\\*\\\*2 as a lower threshold for storing triggers (basically, de-weights triggers with unphysical ratios of Amp33/Amp22 and Amp44/Amp22)
   * - ``interpolate_psd_drift_correction``
     - -
     - False
     - Flag whether to use an interpolated PSD drift correction Default set to preserve search, exposed for debugging purposes
   * - ``return_format``
     - -
     - 'clist'
     - Flag whether to return clist or processedclist
   * - ``adjust_times``
     - -
     - 1
     - Flag to adjust times to assign triggers to the linear-free time of their implied waveforms, used only if return_format==processedclist. The possible values are 0: Adjust the times 1. Adjust the times with the shift of the calpha waveforms, regardless of the kind of waveforms we're using 2: Adjust the times only if using calpha waveforms
   * - ``\*\*psd_drift_kwargs``
     - -
     - -
     - Any extra arguments for recomputing PSD drift

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. clist/processedclist with triggers for the waveform, always a numpy array. Use process_clist to convert to processedclist Note: It adjusts the times to the linear-free ones if calphas is not None, unless adjust_times==False 2. Valid mask on the scores

Docstring
---------

.. code-block:: text

   Generate triggers with sinc-interpolated scores with a single template
   for entire data or a subset of it
   :param wf_whitened_fd:
       Array of 3 x len(rfftfreq(fftsize)) with FFT of waveform
   :param calphas:
       Optional array of calphas representing waveform. If wf_whitened_fd
       is None, we use the calphas to generate a waveform
   :param subset:
       Flag indicating whether to filter the waveform with the entire
       data or subset of it
   :param apply_threshold:
       Flag to apply threshold on chi^2 when collecting triggers
   :param zero_invalid_before_sinc_interp:
       Flag indicating whether to zero invalid scores before
       sinc-interpolating. Default set to preserve search, exposed for
       debugging purposes
   :param support: Support of sinc-interpolating filter
   :param ensure_sinc_support:
       When sinc-interpolating inside an interval, ensure that we only
       retain scores within that interval. Set to False when collecting
       triggers to preserve older runs, use True when optimizing
   :param recompute_psd_drift_correction:
       Flag indicating whether to redefine PSD drift correction, exposed
       for debugging with astrophysical waveforms
   :param interpolate_psd_drift_correction:
       Flag whether to use an interpolated PSD drift correction
       Default set to preserve search, exposed for debugging purposes
   :param marginalized_score_HM: Use a semi-marginalized score instead of 
       |Z22|**2+|Z33|**2+|Z44|**2 as a lower threshold for storing triggers
       (basically, de-weights triggers with unphysical ratios of Amp33/Amp22
        and Amp44/Amp22)
   :param return_format: Flag whether to return clist or processedclist
   :param adjust_times:
       Flag to adjust times to assign triggers to the linear-free time
       of their implied waveforms, used only if
       return_format==processedclist. The possible values are
       0: Adjust the times
       1. Adjust the times with the shift of the calpha waveforms,
          regardless of the kind of waveforms we're using
       2: Adjust the times only if using calpha waveforms
   :param psd_drift_kwargs: Any extra arguments for recomputing PSD drift
   :return:
       1. clist/processedclist with triggers for the waveform, always a
          numpy array. Use process_clist to convert to processedclist
          Note: It adjusts the times to the linear-free ones if calphas
          is not None, unless adjust_times==False
       2. Valid mask on the scores
