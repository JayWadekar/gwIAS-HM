data_operations.calculate_sine_gaussian_overlaps
================================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Returns sine gaussian scores

Signature
---------

.. code-block:: python

   def calculate_sine_gaussian_overlaps(strain_wt, dt, fc_sg, df_sg, freqs_lines = None, mask_freqs = None, fftsize = params.DEF_FFTSIZE)

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
   * - ``dt``
     - -
     - -
     - Time interval between successive elements of data (s)
   * - ``fc_sg``
     - -
     - -
     - Central frequency of sine-Gaussian band (Hz)
   * - ``df_sg``
     - -
     - -
     - df = Upper - lower frequency of sine-Gaussian (Hz)
   * - ``freqs_lines``
     - -
     - None
     - Frequencies for line identification
   * - ``mask_freqs``
     - -
     - None
     - Mask on freqs with zeros at varying lines
   * - ``fftsize``
     - -
     - params.DEF_FFTSIZE
     - FFTsize for overlap-save

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Abs of sine-gaussian overlaps 2. Half support of sine-gaussian filter 3. Cosine overlaps 4. Sine overlaps

Docstring
---------

.. code-block:: text

   Returns sine gaussian scores
   :param strain_wt: Whitened strain data
   :param dt: Time interval between successive elements of data (s)
   :param fc_sg: Central frequency of sine-Gaussian band (Hz)
   :param df_sg: df = Upper - lower frequency of sine-Gaussian (Hz)
   :param freqs_lines: Frequencies for line identification
   :param mask_freqs: Mask on freqs with zeros at varying lines
   :param fftsize: FFTsize for overlap-save
   :return:
       1. Abs of sine-gaussian overlaps
       2. Half support of sine-gaussian filter
       3. Cosine overlaps
       4. Sine overlaps
