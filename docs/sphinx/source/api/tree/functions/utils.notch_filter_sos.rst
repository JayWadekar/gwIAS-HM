utils.notch_filter_sos
======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Defines set of sos filters to apply to notch out lines

Signature
---------

.. code-block:: python

   def notch_filter_sos(dt, freqs, mask_freqs, flow = None, fhigh = None, dfmin_notch = None, notch_pars_in = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dt``
     - -
     - -
     - Time interval between successive elements of data (s)
   * - ``freqs``
     - -
     - -
     - Regular array with frequencies for line identification
   * - ``mask_freqs``
     - -
     - -
     - Mask on freqs with zeros at lines
   * - ``flow``
     - -
     - None
     - Apply notch filters only on frequencies f > flow
   * - ``fhigh``
     - -
     - None
     - Apply notch filters only on frequencies f < fhigh
   * - ``dfmin_notch``
     - -
     - None
     - Minimum frequency width (Hz) that each notch should have
   * - ``notch_pars_in``
     - -
     - None
     - If we already have a list of notch parameters, pass it in to append to and return

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - List of 2-tuples with sos coefficients, and impulse response lengths, to apply to data

Docstring
---------

.. code-block:: text

   Defines set of sos filters to apply to notch out lines
   :param dt: Time interval between successive elements of data (s)
   :param freqs: Regular array with frequencies for line identification
   :param mask_freqs: Mask on freqs with zeros at lines
   :param flow: Apply notch filters only on frequencies f > flow
   :param fhigh: Apply notch filters only on frequencies f < fhigh
   :param dfmin_notch: Minimum frequency width (Hz) that each notch should have
   :param notch_pars_in:
       If we already have a list of notch parameters, pass it in to append to
       and return
   :return:
       List of 2-tuples with sos coefficients, and impulse response lengths,
       to apply to data
