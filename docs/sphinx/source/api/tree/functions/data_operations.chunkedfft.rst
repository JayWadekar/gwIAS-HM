data_operations.chunkedfft
==========================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Returns FFTs of chunks of data for overlap-save. Coordinates assumed dimensionless, as in DFT

Signature
---------

.. code-block:: python

   def chunkedfft(data, fftsize, wl, padmode = 'left', wraparound = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``data``
     - -
     - -
     - Array with the input data (time domain)
   * - ``fftsize``
     - -
     - -
     - Size of FFT for each sub-chunk
   * - ``wl``
     - -
     - -
     - Length of window (time domain)
   * - ``padmode``
     - -
     - 'left'
     - How to pad the data (useful for various choices of window weights and time conventions)
   * - ``wraparound``
     - -
     - True
     - Flag indicating whether to copy data into the padded part to mimic a full FFT

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_chunk x nrfft(fftsize) array of rFFTs of chunks

Docstring
---------

.. code-block:: text

   Returns FFTs of chunks of data for overlap-save. Coordinates assumed
   dimensionless, as in DFT
   :param data: Array with the input data (time domain)
   :param fftsize: Size of FFT for each sub-chunk
   :param wl: Length of window (time domain)
   :param padmode:
       How to pad the data (useful for various choices of window weights and
       time conventions)
   :param wraparound:
       Flag indicating whether to copy data into the padded part to mimic a
       full FFT
   :return: n_chunk x nrfft(fftsize) array of rFFTs of chunks
