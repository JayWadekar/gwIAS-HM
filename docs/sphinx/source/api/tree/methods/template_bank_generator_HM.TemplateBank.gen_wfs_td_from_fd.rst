template_bank_generator_HM.TemplateBank.gen_wfs_td_from_fd
==========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Generates conditioned time-domain waveforms given frequency domain waveforms.

Signature
---------

.. code-block:: python

   def gen_wfs_td_from_fd(self, wfs_fd_original, dt_extra = 0, whiten = False, highpass = True, truncate = False, target_snr = None, fs_in = None, log = False, **conditioning_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wfs_fd_original``
     - -
     - -
     - ... x fs_in array with frequency domain waveforms (can be a vector for n_wf = 1)
   * - ``dt_extra``
     - -
     - 0
     - Increase roll by this time for safety (s)
   * - ``whiten``
     - -
     - False
     - Flag whether to return the whitened waveform
   * - ``highpass``
     - -
     - True
     - Flag indicating whether to high-pass filter the waveform
   * - ``truncate``
     - -
     - False
     - Flag indicating whether to truncate the waveform. Ensure that the wf is linear-free before truncating.
   * - ``target_snr``
     - -
     - None
     - Target SNR to enforce, pass None to not apply this If this is a scalar, then all waveforms in wfs_fd have this SNR If this is an array of shape wfs_fd.shape[:-1], then the individual waveforms have the desired SNRs If whiten == True, this is the norm of the truncated and possibly highpassed whitened waveform If whiten == False and truncate == True or highpass == True, there are losses associated with conditioning
   * - ``fs_in``
     - -
     - None
     - Array of input frequencies. None indicates fs_in = self.fs_fft
   * - ``log``
     - -
     - False
     - Flag indicating whether we passed the log of the waveforms in
   * - ``\*\*conditioning_kwargs``
     - -
     - -
     - If known, pass conditioning parameters (useful when whitening different waveforms), defaults to those in the bank Can use multiple normfacs if needed

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - ... x fftsize array of TD waveforms (can be vector for n_wf=1)

Docstring
---------

.. code-block:: text

   Generates conditioned time-domain waveforms given frequency domain
   waveforms.
   :param wfs_fd_original:
       ... x fs_in array with frequency domain waveforms
       (can be a vector for n_wf = 1)
   :param dt_extra: Increase roll by this time for safety (s)
   :param whiten: Flag whether to return the whitened waveform
   :param highpass:
       Flag indicating whether to high-pass filter the waveform
   :param truncate:
       Flag indicating whether to truncate the waveform.
       Ensure that the wf is linear-free before truncating.
   :param target_snr:
       Target SNR to enforce, pass None to not apply this
       If this is a scalar, then all waveforms in wfs_fd have this SNR
       If this is an array of shape wfs_fd.shape[:-1], then the individual
       waveforms have the desired SNRs
       If whiten == True, this is the norm of the truncated and possibly
       highpassed whitened waveform
       If whiten == False and truncate == True or highpass == True, there
       are losses associated with conditioning
   :param fs_in:
       Array of input frequencies. None indicates fs_in = self.fs_fft
   :param log:
       Flag indicating whether we passed the log of the waveforms in
   :param conditioning_kwargs:
       If known, pass conditioning parameters (useful when whitening
       different waveforms), defaults to those in the bank
       Can use multiple normfacs if needed
   :return:
       ... x fftsize array of TD waveforms (can be vector for n_wf=1)
