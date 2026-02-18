triggers_single_detector_HM.TriggerList.hole_snr_correction
===========================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Computes SNR loss due to the mask (with appropriately inpainted data) Uses the stationary phase approximation, so valid only when the frequencies >> 1/hole size. We do not assume waveforms are normalized. Note: extended by Jay to include corrections due to bands in mask_stft

Signature
---------

.. code-block:: python

   def hole_snr_correction(wfs_whitened_fd, chunked_mask_f, fftsize, support_whitened_wf, cheat_overlap_save = False, only_22 = False, erase_bands = False, **erase_bands_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wfs_whitened_fd``
     - -
     - -
     - nwf x 3 x length rfft(fftsize) complex array with frequency domain waveforms (can be vector for nwf = 1) Convention: power is towards the right side
   * - ``chunked_mask_f``
     - -
     - -
     - Chunked FFT of mask
   * - ``fftsize``
     - -
     - -
     - FFTsize for chunked mask
   * - ``support_whitened_wf``
     - -
     - -
     - Support of whitened waveform
   * - ``cheat_overlap_save``
     - -
     - False
     - Flag indicating to not use support_whitened_wf in overlap-save
   * - ``only_22``
     - -
     - False
     - Use only 22 mode instead of the 3 modes everywhere
   * - ``erase_bands``
     - -
     - False
     - Flag to erase bands in time-frequency space
   * - ``\*\*erase_bands_kwargs``
     - -
     - -
     - If erase_bands, pass corresponding kwargs: mask_stft, noverlaps, dt

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - nwf x 3 x (nchunk x chunksize) array with hole corrections (use first len(data) values in each row) Can be vector if wf_whitened_fd is a vector

Docstring
---------

.. code-block:: text

   Computes SNR loss due to the mask (with appropriately inpainted data)
   Uses the stationary phase approximation, so valid only when the
   frequencies >> 1/hole size. We do not assume waveforms are normalized.
   Note: extended by Jay to include corrections due to bands in mask_stft
   :param wfs_whitened_fd:
       nwf x 3 x length rfft(fftsize) complex array with frequency domain
       waveforms (can be vector for nwf = 1)
       Convention: power is towards the right side
   :param chunked_mask_f: Chunked FFT of mask
   :param fftsize: FFTsize for chunked mask
   :param support_whitened_wf: Support of whitened waveform
   :param cheat_overlap_save:
       Flag indicating to not use support_whitened_wf in overlap-save
   :param only_22: Use only 22 mode instead of the 3 modes everywhere
   :param erase_bands: Flag to erase bands in time-frequency space
   :param erase_bands_kwargs:
       If erase_bands, pass corresponding kwargs: mask_stft, noverlaps, dt
   :return: nwf x 3 x (nchunk x chunksize) array with hole corrections
            (use first len(data) values in each row)
            Can be vector if wf_whitened_fd is a vector
