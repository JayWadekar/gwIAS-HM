template_bank_generator_HM.TemplateBank.gen_wf_td_from_pars
===========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

WARNING: Not modified by Jay for the case of HM

Signature
---------

.. code-block:: python

   def gen_wf_td_from_pars(self, dt = None, target_snr = None, highpass = False, phase = 0, **pars)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dt``
     - -
     - None
     - Time spacing of samples (s). If None, uses self.dt The following variables need bank to be conditioned
   * - ``target_snr``
     - -
     - None
     - Target SNR to achieve. Pass None to not apply this
   * - ``highpass``
     - -
     - False
     - Flag indicating whether to high-pass filter waveforms
   * - ``phase``
     - -
     - 0
     - Phase of waveform
   * - ``\*\*pars``
     - -
     - -
     - Desired subset of m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, l1, l2, approximant. If approximant is None, uses self.approximant. See ComputeWF_farray.genwf_td for other defaults

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Array with time-domain waveform (with spacing dt)

Docstring
---------

.. code-block:: text

   WARNING: Not modified by Jay for the case of HM
   
   Generate time domain waveform for injection, let lal do the
   conditioning for us (works only for approximants with td waveforms)
   :param dt: Time spacing of samples (s). If None, uses self.dt
   The following variables need bank to be conditioned
   :param target_snr: Target SNR to achieve. Pass None to not apply this
   :param highpass: Flag indicating whether to high-pass filter waveforms
   :param phase: Phase of waveform
   :param pars:
       Desired subset of m1, m2, s1x, s1y, s1z, s2x, s2y,
       s2z, l1, l2, approximant. If approximant is None, uses
       self.approximant. See ComputeWF_farray.genwf_td for other defaults
   :return: Array with time-domain waveform (with spacing dt)
