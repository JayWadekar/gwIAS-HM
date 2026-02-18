template_bank_generator_HM.TemplateBank.set_waveform_conditioning
=================================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Finds normalization factor, support, and shift for whitened waveforms in bank if they aren't already defined \\\*NOTE\\\* for PSD estimation, need: fftsize \\\* dt > chunktime

Signature
---------

.. code-block:: python

   def set_waveform_conditioning(self, fftsize, dt, wt_filter_fd = None, min_support = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fftsize``
     - -
     - -
     - (int) Number of samples in FFT for the template
   * - ``dt``
     - -
     - -
     - (float) Sampling interval of template (in seconds)
   * - ``wt_filter_fd``
     - -
     - None
     - Frequency domain whitening filter (~irfft of 1/ASD). Lives on rfftfreq(fftsize, dt). If None, defaults to version in bank
   * - ``min_support``
     - -
     - None
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
     - Updates 1. self.fftsize 2. self.dt 3. self.wt_filter_fd 4. self.support_whitened_wf: TD support of waveform (in units of dt). Hardcoded to params.DEF_MAX_WFDURATION 5. self.shift_wf: Shift applied for weight on the right of filter (in units of dt) 6. self.normfac: Normalization factor to divide waveform \\\* whitening filter by for convolution with whitened data and filter parameters

Docstring
---------

.. code-block:: text

   Finds normalization factor, support, and shift for whitened waveforms
   in bank if they aren't already defined
   *NOTE* for PSD estimation, need:  fftsize * dt > chunktime
   :param fftsize: (int) Number of samples in FFT for the template
   :param dt: (float) Sampling interval of template (in seconds)
   :param wt_filter_fd:
       Frequency domain whitening filter (~irfft of 1/ASD). Lives on
       rfftfreq(fftsize, dt). If None, defaults to version in bank
   :return: Updates
            1. self.fftsize
            2. self.dt
            3. self.wt_filter_fd
            4. self.support_whitened_wf:
                   TD support of waveform (in units of dt). Hardcoded to
                   params.DEF_MAX_WFDURATION
            5. self.shift_wf:
                   Shift applied for weight on the right of filter
                   (in units of dt)
            6. self.normfac:
                   Normalization factor to divide waveform *
                   whitening filter by for convolution with whitened data
            and filter parameters
