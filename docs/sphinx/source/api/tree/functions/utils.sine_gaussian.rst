utils.sine_gaussian
===================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Function to return time-domain sine-gaussian pulses (ready for FFT)

Signature
---------

.. code-block:: python

   def sine_gaussian(dt, fc, df)

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
   * - ``fc``
     - -
     - -
     - Central frequency of transient (Hz)
   * - ``df``
     - -
     - -
     - Spread in frequency, high - low (Hz)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Time-domain cosine pulse 2. Time-domain sine pulse, both satisfying sum_t pulse^2 = 1 3. Support of pulse (has 2 \\\* support - 1 nonzero entries)

Docstring
---------

.. code-block:: text

   Function to return time-domain sine-gaussian pulses (ready for FFT)
   :param dt: Sampling interval (s)
   :param fc: Central frequency of transient (Hz)
   :param df: Spread in frequency, high - low (Hz)
   :return: 1. Time-domain cosine pulse
            2. Time-domain sine pulse, both satisfying sum_t pulse^2 = 1
            3. Support of pulse (has 2 * support - 1 nonzero entries)
