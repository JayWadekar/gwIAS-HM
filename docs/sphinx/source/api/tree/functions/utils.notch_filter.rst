utils.notch_filter
==================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Creates notch filter

Signature
---------

.. code-block:: python

   def notch_filter(dt, fc, df)

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
     - Central frequency of notch (Hz)
   * - ``df``
     - -
     - -
     - -3dB bandwidth in frequency, high - low (Hz)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Numerator and denominator coeffs of notch filter

Docstring
---------

.. code-block:: text

   Creates notch filter
   :param dt: Sampling interval (s)
   :param fc: Central frequency of notch (Hz)
   :param df: -3dB bandwidth in frequency, high - low (Hz)
   :return: Numerator and denominator coeffs of notch filter
