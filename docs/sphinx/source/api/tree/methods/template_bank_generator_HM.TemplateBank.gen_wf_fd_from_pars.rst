template_bank_generator_HM.TemplateBank.gen_wf_fd_from_pars
===========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Convenience interface to gen_waveform that removes the linear component of the phase in the same way as was used to make the bank

Signature
---------

.. code-block:: python

   def gen_wf_fd_from_pars(self, fs_out = None, gen_domain = 'fd', phase = 0, linear_free = True, **pars)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fs_out``
     - -
     - None
     - If desired, generate waveform at these frequencies. Defaults to self.fs_fft (useful for, e.g., relative binning. In case we're passing fs_out, make sure that unwrap errors won't occur)
   * - ``gen_domain``
     - -
     - 'fd'
     - String indicating whether the approximant is td or fd, use 'td' when LAL doesn't have fd version implemented
   * - ``phase``
     - -
     - 0
     - Phase of waveform
   * - ``linear_free``
     - -
     - True
     - Flag whether to return the linear free waveform
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
     - Array of length self.fs_fft with unconditioned frequency domain waveform

Docstring
---------

.. code-block:: text

   Convenience interface to gen_waveform that removes the linear
   component of the phase in the same way as was used to make the bank
   :param fs_out:
       If desired, generate waveform at these frequencies. Defaults to
       self.fs_fft (useful for, e.g., relative binning. In case we're
       passing fs_out, make sure that unwrap errors won't occur)
   :param gen_domain:
       String indicating whether the approximant is td or fd, use 'td'
       when LAL doesn't have fd version implemented
   :param phase: Phase of waveform
   :param linear_free: Flag whether to return the linear free waveform
   :param pars: Desired subset of m1, m2, s1x, s1y, s1z, s2x, s2y,
       s2z, l1, l2, approximant. If approximant is None, uses
       self.approximant. See ComputeWF_farray.genwf_td for other defaults
   :return:
       Array of length self.fs_fft with unconditioned frequency
       domain waveform
