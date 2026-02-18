utils.band_filter
=================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Creates desired filter, and computes its impulse response length

Signature
---------

.. code-block:: python

   def band_filter(dt, fmin = None, fmax = None, wfac = None, btype = 'bandpass', order = params.ORDER, filter_type = 'butter')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dt``
     - -
     - -
     - Sampling interval (s)
   * - ``fmin``
     - -
     - None
     - Lower critical frequency (-3dB point). Pass None for lowpass
   * - ``fmax``
     - -
     - None
     - Higher critical frequency (-3dB point). Pass None for highpass
   * - ``wfac``
     - -
     - None
     - Fraction of impulse response to capture. If None, we use defaults from params
   * - ``btype``
     - -
     - 'bandpass'
     - Kind of filter, should be either 'bandpass', 'bandstop', 'high', or 'low'
   * - ``order``
     - -
     - params.ORDER
     - Order of filter
   * - ``filter_type``
     - -
     - 'butter'
     - Kind of filter to create

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - sos representation of filter coefficients, impulse response length

Docstring
---------

.. code-block:: text

   Creates desired filter, and computes its impulse response length
   :param dt: Sampling interval (s)
   :param fmin: Lower critical frequency (-3dB point). Pass None for lowpass
   :param fmax: Higher critical frequency (-3dB point). Pass None for highpass
   :param wfac: Fraction of impulse response to capture. If None, we use
                defaults from params
   :param btype: Kind of filter, should be either 'bandpass', 'bandstop',
                 'high', or 'low'
   :param order: Order of filter
   :param filter_type: Kind of filter to create
   :return: sos representation of filter coefficients, impulse response length
