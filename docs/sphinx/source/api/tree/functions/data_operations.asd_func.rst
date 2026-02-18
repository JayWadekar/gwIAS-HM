data_operations.asd_func
========================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Returns function that takes frequencies in Hz and gives the interpolated ASD in units of 1/Hz^0.5

Signature
---------

.. code-block:: python

   def asd_func(fint, psdint, fmin = None, fmax = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fint``
     - -
     - -
     - Array with list of sampled frequencies
   * - ``psdint``
     - -
     - -
     - Array with list of computed PSDs
   * - ``fmin``
     - -
     - None
     - Blow up PSDs below fmin due to lack of calibration (None if no fmin)
   * - ``fmax``
     - -
     - None
     - Blow up PSDs above fmax to avoid features in the measured PSD (Pass None if no fmax)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Function that takes frequencies and returns ASDs by linearly interpolating the PSDs

Docstring
---------

.. code-block:: text

   Returns function that takes frequencies in Hz and gives the interpolated
   ASD in units of 1/Hz^0.5
   :param fint: Array with list of sampled frequencies
   :param psdint: Array with list of computed PSDs
   :param fmin: Blow up PSDs below fmin due to lack of calibration (None if no
                fmin)
   :param fmax: Blow up PSDs above fmax to avoid features in the measured PSD
                (Pass None if no fmax)
   :return: Function that takes frequencies and returns ASDs by linearly
            interpolating the PSDs
