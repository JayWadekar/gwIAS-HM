data_operations.norm_matched_filter_overlap
===========================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Computes matched filter overlap for a sliding template. Coordinates assumed dimensionless, as in DFT.

Signature
---------

.. code-block:: python

   def norm_matched_filter_overlap(chunked_data_f, wf_whitened_fd, fftsize, support_wf)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``chunked_data_f``
     - -
     - -
     - n_chunk x rfftsize(fftsize) array with data in frequency domain
   * - ``wf_whitened_fd``
     - -
     - -
     - Frequency domain template (length rfft(fftsize)) Convention: power is towards the right side
   * - ``fftsize``
     - -
     - -
     - Size of FFT for each sub-chunk
   * - ``support_wf``
     - -
     - -
     - Time domain support for whitened waveform template

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Array of length n_chunk x chunksize with cosine and sine components of matched filtering overlap

Docstring
---------

.. code-block:: text

   Computes matched filter overlap for a sliding template. Coordinates
   assumed dimensionless, as in DFT. Returns complex number whose real and
   imaginary parts are N(0,1)
   :param chunked_data_f: n_chunk x rfftsize(fftsize) array with data in
                          frequency domain
   :param wf_whitened_fd: Frequency domain template (length rfft(fftsize))
                          Convention: power is towards the right side
   :param fftsize: Size of FFT for each sub-chunk
   :param support_wf: Time domain support for whitened waveform template
   :return: Array of length n_chunk x chunksize with cosine and sine components
            of matched filtering overlap
