triggers_single_detector_HM.TriggerList.scores_wf
=================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Computes overlaps and hole corrections for multiple waveforms We do not assume waveforms are normalized

Signature
---------

.. code-block:: python

   def scores_wf(wfs_whitened_fd, chunked_data_f, chunked_mask_f, valid_mask, fftsize, support_whitened_wf, cheat_overlap_save = False, zero_invalid = True, only_22 = False, erase_bands = False, **erase_bands_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wfs_whitened_fd``
     - -
     - -
     - nwf x 3 x length rfft(fftsize) complex array with frequency domain waveforms Convention: power is towards the right side
   * - ``chunked_data_f``
     - -
     - -
     - Chunked FFT of data
   * - ``chunked_mask_f``
     - -
     - -
     - Chunked FFT of mask
   * - ``valid_mask``
     - -
     - -
     - Valid mask
   * - ``fftsize``
     - -
     - -
     - FFTsize for chunked data and mask
   * - ``support_whitened_wf``
     - -
     - -
     - Support of whitened waveform
   * - ``cheat_overlap_save``
     - -
     - False
     - Flag indicating to not use support_whitened_wf in overlap-save
   * - ``zero_invalid``
     - -
     - True
     - Flag indicating whether to zero invalid scores
   * - ``only_22``
     - -
     - False
     - Only use 22 instead of the 3 modes everywhere
   * - ``erase_bands``
     - -
     - False
     - Flag to erase bands in time-frequency space
   * - ``\*\*erase_bands_kwargs``
     - -
     - -
     - If erase_bands, pass corresponding kwargs (mask_stft, noverlaps, dt)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. nwf x 3 x (nchunk x chunksize) array with overlaps (zero where the hole correction cannot be trusted) 2. nwf x 3 x (nchunk x chunksize) array with hole corrections 3. nwf x 3 x (nchunk x chunksize) Boolean array indicating validity of overlaps Use first len(data) values of both, can be vectors if wf_whitened_fd is a vector

Docstring
---------

.. code-block:: text

   Computes overlaps and hole corrections for multiple waveforms
   We do not assume waveforms are normalized
   :param wfs_whitened_fd:
       nwf x 3 x length rfft(fftsize) complex array with frequency domain
       waveforms
       Convention: power is towards the right side
   :param chunked_data_f: Chunked FFT of data
   :param chunked_mask_f: Chunked FFT of mask
   :param valid_mask: Valid mask
   :param fftsize: FFTsize for chunked data and mask
   :param support_whitened_wf: Support of whitened waveform
   :param cheat_overlap_save:
       Flag indicating to not use support_whitened_wf in overlap-save
   :param zero_invalid: Flag indicating whether to zero invalid scores
   :param only_22: Only use 22 instead of the 3 modes everywhere
   :param erase_bands: Flag to erase bands in time-frequency space
   :param erase_bands_kwargs:
       If erase_bands, pass corresponding kwargs (mask_stft, noverlaps, dt)
   :return:
       1. nwf x 3 x (nchunk x chunksize) array with overlaps
          (zero where the hole correction cannot be trusted)
       2. nwf x 3 x (nchunk x chunksize) array with hole corrections
       3. nwf x 3 x (nchunk x chunksize) Boolean array indicating validity of
          overlaps
       Use first len(data) values of both, can be vectors if wf_whitened_fd
       is a vector
