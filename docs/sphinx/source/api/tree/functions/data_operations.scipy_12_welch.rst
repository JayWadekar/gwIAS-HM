data_operations.scipy_12_welch
==============================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Cheating to copy Scipy 1.2's Welch method, to access the average attribute Note: Assumes fs is an integer in converting times to mask indices

Signature
---------

.. code-block:: python

   def scipy_12_welch(x, y = None, fs = 1.0, window = 'hann', nperseg = None, noverlap = None, nfft = None, detrend = 'constant', return_onesided = True, scaling = 'density', axis = -1, average = 'mean', mask = None, line_id_ver = 'old')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``x``
     - -
     - -
     - -
   * - ``y``
     - -
     - None
     - -
   * - ``fs``
     - -
     - 1.0
     - -
   * - ``window``
     - -
     - 'hann'
     - -
   * - ``nperseg``
     - -
     - None
     - -
   * - ``noverlap``
     - -
     - None
     - -
   * - ``nfft``
     - -
     - None
     - -
   * - ``detrend``
     - -
     - 'constant'
     - -
   * - ``return_onesided``
     - -
     - True
     - -
   * - ``scaling``
     - -
     - 'density'
     - -
   * - ``axis``
     - -
     - -1
     - -
   * - ``average``
     - -
     - 'mean'
     - -
   * - ``mask``
     - -
     - None
     - -
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
     - 1. Array with frequency axis (Hz) 2. Array with one-sided PSD (Hz^-1) 3. Crude line mask with lines zeroed 4. Loud line mask with lines zeroed

Docstring
---------

.. code-block:: text

   Cheating to copy Scipy 1.2's Welch method, to access the average
   attribute
   Note: Assumes fs is an integer in converting times to mask indices
   :param line_id_ver:
       Flag to use old or new version of line-identification code
   :return: 1. Array with frequency axis (Hz)
            2. Array with one-sided PSD (Hz^-1)
            3. Crude line mask with lines zeroed
            4. Loud line mask with lines zeroed
