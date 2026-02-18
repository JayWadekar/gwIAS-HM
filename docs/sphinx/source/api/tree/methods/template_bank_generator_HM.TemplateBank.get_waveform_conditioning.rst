template_bank_generator_HM.TemplateBank.get_waveform_conditioning
=================================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Useful function to get conditioning parameters for whitened waveforms

Signature
---------

.. code-block:: python

   def get_waveform_conditioning(fftsize, dt, whitened_wf_fd)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fftsize``
     - -
     - -
     - Number of time-domain indices
   * - ``dt``
     - -
     - -
     - Time domain step size (s)
   * - ``whitened_wf_fd``
     - -
     - -
     - len(rfftfreq(fftsize)) array with FD whitened waveform IMP note: The wf should be linear free.

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Support of waveform in units of indices 2. Shift to put waveform with weight toward the right side 3. Normalization factor

Docstring
---------

.. code-block:: text

   Useful function to get conditioning parameters for whitened waveforms
   :param fftsize: Number of time-domain indices
   :param dt: Time domain step size (s)
   :param whitened_wf_fd:
       len(rfftfreq(fftsize)) array with FD whitened waveform
       IMP note: The wf should be linear free.
   :return: 1. Support of waveform in units of indices
            2. Shift to put waveform with weight toward the right side
            3. Normalization factor
