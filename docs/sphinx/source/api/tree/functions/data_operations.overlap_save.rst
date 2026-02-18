data_operations.overlap_save
============================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Convolution with the overlap-save method. Look at the modes in chunkedfft to understand what transients are captured at the edges Coordinates assumed dimensionless, as in DFT Note: Overlap-save is only justified when the window is strictly compact and contained within wl

Signature
---------

.. code-block:: python

   def overlap_save(chunked_data_f, window_f, fftsize, wl)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``chunked_data_f``
     - -
     - -
     - n_chunk x nrfft(fftsize) array of rFFTs of data chunks
   * - ``window_f``
     - -
     - -
     - Complex array of length nrfft(fftsize) RFFT of {window with support within wl, padded to fftsize in the time domain}
   * - ``fftsize``
     - -
     - -
     - Size of FFT for each sub-chunk
   * - ``wl``
     - -
     - -
     - Length of window (time domain)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Array of length nchunk x chunksize with convolved data

Docstring
---------

.. code-block:: text

   Convolution with the overlap-save method. Look at the modes in chunkedfft
   to understand what transients are captured at the edges
   Coordinates assumed dimensionless, as in DFT
   Note: Overlap-save is only justified when the window is strictly compact
   and contained within wl
   :param chunked_data_f: n_chunk x nrfft(fftsize) array of rFFTs of data chunks
   :param window_f: Complex array of length nrfft(fftsize) RFFT of {window with
                    support within wl, padded to fftsize in the time domain}
   :param fftsize: Size of FFT for each sub-chunk
   :param wl: Length of window (time domain)
   :return: Array of length nchunk x chunksize with convolved data
