template_bank_generator_HM.compute_hplus_hcross
===============================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Generate frequency domain waveform using LAL. Return hplus, hcross evaluated at f.

Signature
---------

.. code-block:: python

   def compute_hplus_hcross(f, par_dic, approximant: str, harmonic_modes = None, force_nnlo = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``f``
     - Frequency array in Hz
     - -
     - -
   * - ``par_dic``
     - Dictionary of source parameters. Needs to have these keys:
     - -
     - \\\* m1, m2: component masses (Msun) \\\* d_luminosity: luminosity distance (Mpc) \\\* iota: inclination (rad) \\\* phi_ref: phase at reference frequency (rad) \\\* f_ref: reference frequency (Hz) \\\* s1x, s1y, s1z, s2x, s2y, s2z: dimensionless spins \\\* l1, l2: dimensionless tidal deformabilities
   * - ``approximant``
     - String with the approximant name.
     - -
     - -
   * - ``harmonic_modes``
     - Optional, list of 2-tuples with (l, m) pairs
     - None
     - specifying which (co-precessing frame) higher-order modes to include.
   * - ``force_nnlo``
     - -
     - True
     - -

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - -

Docstring
---------

.. code-block:: text

   Generate frequency domain waveform using LAL.
   Return hplus, hcross evaluated at f.
   
   Parameters
   ----------
   approximant: String with the approximant name.
   f: Frequency array in Hz
   par_dic: Dictionary of source parameters. Needs to have these keys:
                * m1, m2: component masses (Msun)
                * d_luminosity: luminosity distance (Mpc)
                * iota: inclination (rad)
                * phi_ref: phase at reference frequency (rad)
                * f_ref: reference frequency (Hz)
                * s1x, s1y, s1z, s2x, s2y, s2z: dimensionless spins
                * l1, l2: dimensionless tidal deformabilities
   harmonic_modes: Optional, list of 2-tuples with (l, m) pairs
                 specifying which (co-precessing frame) higher-order
                 modes to include.
