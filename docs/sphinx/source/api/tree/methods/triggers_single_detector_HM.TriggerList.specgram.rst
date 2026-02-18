triggers_single_detector_HM.TriggerList.specgram
================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Makes spectrogram of whitened strain data

Signature
---------

.. code-block:: python

   def specgram(self, nfft = None, tmin = None, tmax = None, t0 = 0, ax = None, vmin = None, vmax = None, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``nfft``
     - -
     - None
     - Number of data points to use per FFT, defaults to number per second
   * - ``tmin``
     - -
     - None
     - Minimum time for spectrogram, relative to t0 (s)
   * - ``tmax``
     - -
     - None
     - Minimum time for spectrogram, relative to t0 (s)
   * - ``t0``
     - -
     - 0
     - Origin to measure time relative to, when specifying tmin-tmax
   * - ``ax``
     - -
     - None
     - Axis to place plot in
   * - ``vmin``
     - -
     - None
     - Minimum value for colorbar
   * - ``vmax``
     - -
     - None
     - Maximum value for colorbar
   * - ``\*\*kwargs``
     - -
     - -
     - -

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Axis with spectrogram

Docstring
---------

.. code-block:: text

   Makes spectrogram of whitened strain data
   :param nfft: Number of data points to use per FFT, defaults to number
                per second
   :param tmin: Minimum time for spectrogram, relative to t0 (s)
   :param tmax: Minimum time for spectrogram, relative to t0 (s)
   :param t0: Origin to measure time relative to, when specifying tmin-tmax
   :param vmin: Minimum value for colorbar
   :param vmax: Maximum value for colorbar
   :param ax: Axis to place plot in
   :return: Axis with spectrogram
