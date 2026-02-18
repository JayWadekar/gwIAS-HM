data_operations.data_to_asdfunc
===============================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Passes the data to welch and returns a function that maps frequencies to ASD

Signature
---------

.. code-block:: python

   def data_to_asdfunc(strains, mask, fs, chunktime_psd = params.DEF_CHUNKTIME_PSD, fmax = params.FMAX_PSD, average = 'median', line_id_ver = 'old')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``strains``
     - -
     - -
     - Array with strains
   * - ``mask``
     - -
     - -
     - Boolean quality mask on strain data at 1 Hz
   * - ``fs``
     - -
     - -
     - Sampling frequency (Hz)
   * - ``chunktime_psd``
     - -
     - params.DEF_CHUNKTIME_PSD
     - Chunktime for PSD estimation
   * - ``fmax``
     - -
     - params.FMAX_PSD
     - Maximum frequency to estimate the PSD at
   * - ``average``
     - -
     - 'median'
     - Averaging method to use
   * - ``line_id_ver``
     - -
     - 'old'
     - Flag to use old or new version of line-identification code

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Array with frequencies (Hz) 2. Array with PSD (Hz^-1) 3. Function mapping frequencies to ASD 4. Crude mask on frequencies with lines marked with zeros 5. Mask on frequencies with loud lines marked with zeros

Docstring
---------

.. code-block:: text

   Passes the data to welch and returns a function that maps frequencies to ASD
   :param strains: Array with strains
   :param mask: Boolean quality mask on strain data at 1 Hz
   :param fs: Sampling frequency (Hz)
   :param chunktime_psd: Chunktime for PSD estimation
   :param fmax: Maximum frequency to estimate the PSD at
   :param average: Averaging method to use
   :param line_id_ver:
       Flag to use old or new version of line-identification code
   :return: 1. Array with frequencies (Hz)
            2. Array with PSD (Hz^-1)
            3. Function mapping frequencies to ASD
            4. Crude mask on frequencies with lines marked with zeros
            5. Mask on frequencies with loud lines marked with zeros
