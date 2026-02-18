data_operations.find_whitened_outliers
======================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Zeros mask in place on either side of outliers in whitened data stream

Signature
---------

.. code-block:: python

   def find_whitened_outliers(strain_wt, qmask, valid_mask, dt, clipwidth, sigma_clipping_threshold = None, outlier_fraction = params.OUTLIER_FRAC, zero_mask = True, nfire = params.NPERFILE, renorm_wt = True, sc_n01 = 1.0, mask_save = None, verbose = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``strain_wt``
     - -
     - -
     - Whitened strain data
   * - ``qmask``
     - -
     - -
     - Boolean mask with zeros at holes in unwhitened data
   * - ``valid_mask``
     - -
     - -
     - Boolean mask with zeros where we cannot trust whitened data
   * - ``dt``
     - -
     - -
     - Time interval between successive elements of strain_wt (s)
   * - ``clipwidth``
     - -
     - -
     - Half-width of window around outliers to zero (in s)
   * - ``sigma_clipping_threshold``
     - -
     - None
     - Threshold for sigma clipping computed from waveform. If none, clipping depends on params.NPERFILE
   * - ``outlier_fraction``
     - -
     - params.OUTLIER_FRAC
     - If outlier is more than 1/outlier_fraction x sigma clipping threshold, adjust threshold to avoid overcorrecting due to ringing of the outlier against the whitening filter
   * - ``zero_mask``
     - -
     - True
     - Flag indicating whether to zero the mask at glitches. Turn off for vetoing
   * - ``nfire``
     - -
     - params.NPERFILE
     - Number of times glitch detector fires per perfect file
   * - ``renorm_wt``
     - -
     - True
     - Flag whether we scaled the whitened data to have unit variance after highpass
   * - ``sc_n01``
     - -
     - 1.0
     - Factor to multiply the whitened data with to get each data point to be N(0,1), used only if renorm_wt is False (which is the default from now)
   * - ``mask_save``
     - -
     - None
     - If desired, boolean mask on strain_wt with zeros at data to avoid
   * - ``verbose``
     - -
     - True
     - If true, prints details about outlier check

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Zeros mask in place, returns 1. List of indices that jumped 2. Mask with zeros where we should zero the data as a result 3. Flag indicating whether we had an overly loud outlier 4. String with details of the test

Docstring
---------

.. code-block:: text

   Zeros mask in place on either side of outliers in whitened data stream
   :param strain_wt: Whitened strain data
   :param qmask: Boolean mask with zeros at holes in unwhitened data
   :param valid_mask:
       Boolean mask with zeros where we cannot trust whitened data
   :param dt: Time interval between successive elements of strain_wt (s)
   :param clipwidth: Half-width of window around outliers to zero (in s)
   :param sigma_clipping_threshold:
       Threshold for sigma clipping computed from waveform. If none, clipping
       depends on params.NPERFILE
   :param outlier_fraction:
       If outlier is more than 1/outlier_fraction x sigma clipping threshold,
       adjust threshold to avoid overcorrecting due to ringing of the outlier
       against the whitening filter
   :param zero_mask:
       Flag indicating whether to zero the mask at glitches. Turn off for
       vetoing
   :param nfire: Number of times glitch detector fires per perfect file
   :param renorm_wt:
       Flag whether we scaled the whitened data to have unit variance after
       highpass
   :param sc_n01:
       Factor to multiply the whitened data with to get each data point to be
       N(0,1), used only if renorm_wt is False (which is the default from now)
   :param mask_save:
       If desired, boolean mask on strain_wt with zeros at data to avoid
   :param verbose: If true, prints details about outlier check
   :return: Zeros mask in place, returns
            1. List of indices that jumped
            2. Mask with zeros where we should zero the data as a result
            3. Flag indicating whether we had an overly loud outlier
            4. String with details of the test
