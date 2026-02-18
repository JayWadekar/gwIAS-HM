template_bank_generator_HM.TemplateBank.gen_random_wfs_td
=========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Has not been updated by Jay. Generates random time-domain waveforms, OK for FFT test but not for injection due to wraparound

Signature
---------

.. code-block:: python

   def gen_random_wfs_td(self, n_wf = 1, highpass = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``n_wf``
     - -
     - 1
     - Number of random waveforms to generate
   * - ``highpass``
     - -
     - True
     - Flag indicating whether to lowpass filter the waveform

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_wf x fftsize array of random TD waveforms (vector for n_wf=1)

Docstring
---------

.. code-block:: text

   Has not been updated by Jay.
   Generates random time-domain waveforms, OK for FFT test but not
   for injection due to wraparound
   :param n_wf: Number of random waveforms to generate
   :param highpass: Flag indicating whether to lowpass filter the waveform
   :return: n_wf x fftsize array of random TD waveforms
            (vector for n_wf=1)
