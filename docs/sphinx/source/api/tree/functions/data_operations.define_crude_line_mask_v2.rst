data_operations.define_crude_line_mask_v2
=========================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Function to flag lines in the PSD, errs on the side of marking lines

Signature
---------

.. code-block:: python

   def define_crude_line_mask_v2(freqs, psds, nindpsd, lw_check = 0.5, fmin = params.FMIN_ANALYSIS, niter = 3)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``freqs``
     - -
     - -
     - Array with frequencies (in Hz)
   * - ``psds``
     - -
     - -
     - Array with PSDs (in 1/Hz)
   * - ``nindpsd``
     - -
     - -
     - Number of independent windows in Welch estimate for the PSD
   * - ``lw_check``
     - -
     - 0.5
     - Parameter for smoothing window to check for lines (in Hz), equal to the assumed width of the lines, as well as roughly 1/3 the width of the frequency interval we smooth over
   * - ``fmin``
     - -
     - params.FMIN_ANALYSIS
     - Look for lines only at f >= fmin
   * - ``niter``
     - -
     - 3
     - Number of passes to find lines

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Boolean array of length freqs with zeros marking lines 2. Boolean array of length freqs with zeros marking loud lines

Docstring
---------

.. code-block:: text

   Function to flag lines in the PSD, errs on the side of marking lines
   :param freqs: Array with frequencies (in Hz)
   :param psds: Array with PSDs (in 1/Hz)
   :param nindpsd: Number of independent windows in Welch estimate for the PSD
   :param lw_check:
       Parameter for smoothing window to check for lines (in Hz), equal to the
       assumed width of the lines, as well as roughly 1/3 the width of the
       frequency interval we smooth over
   :param fmin: Look for lines only at f >= fmin
   :param niter: Number of passes to find lines
   :return: 1. Boolean array of length freqs with zeros marking lines
            2. Boolean array of length freqs with zeros marking loud lines
