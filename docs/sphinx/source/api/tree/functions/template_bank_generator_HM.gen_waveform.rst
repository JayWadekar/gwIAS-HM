template_bank_generator_HM.gen_waveform
=======================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Generates waveforms with given binary parameters

Signature
---------

.. code-block:: python

   def gen_waveform(fs_gen = None, sub_fac = 1, m1 = 30, m2 = 30, s1x = 0, s1y = 0, s1z = 0, s2x = 0, s2y = 0, s2z = 0, l1 = 0, l2 = 0, approximant = 'IMRPhenomD_NRTidal', unwrap = True, phase = 'random', deltaF = None, f_min = None, f_max = None, inclination = None, phiRef = None, f_ref = None, distance = None, **kwargs_pars)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fs_gen``
     - -
     - None
     - Set of frequencies (Hz) (alternatively, pass deltaF etc...)
   * - ``sub_fac``
     - -
     - 1
     - Factor we subsampled the frequencies by to avoid unwrap errors
   * - ``m1``
     - -
     - 30
     - First mass (in M_sun)
   * - ``m2``
     - -
     - 30
     - Second mass (in M_sun)
   * - ``s1x``
     - -
     - 0
     - X component of first spin (dimensionless)
   * - ``s1y``
     - -
     - 0
     - Y component of first spin (dimensionless)
   * - ``s1z``
     - -
     - 0
     - Z component of first spin (dimensionless)
   * - ``s2x``
     - -
     - 0
     - X component of second spin (dimensionless)
   * - ``s2y``
     - -
     - 0
     - Y component of second spin (dimensionless)
   * - ``s2z``
     - -
     - 0
     - Z component of second spin (dimensionless)
   * - ``l1``
     - -
     - 0
     - Tidal deformalbility of first mass
   * - ``l2``
     - -
     - 0
     - Tidal deformalbility of second mass
   * - ``approximant``
     - -
     - 'IMRPhenomD_NRTidal'
     - LAL string for approximant
   * - ``unwrap``
     - -
     - True
     - Flag indicating whether to unwrap and return amp and phase
   * - ``phase``
     - -
     - 'random'
     - If known, pass phase to use, else pass "random" to indicate a random phase
   * - ``deltaF``
     - -
     - None
     - If on a regular grid, frequency spacing (Hz)
   * - ``f_min``
     - -
     - None
     - If on a regular grid, minimum frequency to query the approximant at (Hz)
   * - ``f_max``
     - -
     - None
     - If on a regular grid, maximum frequency (Hz, ~Nyquist frequency)
   * - ``inclination``
     - -
     - None
     - -
   * - ``phiRef``
     - -
     - None
     - -
   * - ``f_ref``
     - -
     - None
     - -
   * - ``distance``
     - -
     - None
     - -
   * - ``\*\*kwargs_pars``
     - -
     - -
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
     - 1. len(fs_gen / sub_fac) array with amplitude of frequency domain waveform (units of 1/Hz) 2. len(fs_gen / sub_fac) array with phase of frequency domain waveform

Docstring
---------

.. code-block:: text

   Generates waveforms with given binary parameters
   :param fs_gen: Set of frequencies (Hz) (alternatively, pass deltaF etc...)
   :param sub_fac:
       Factor we subsampled the frequencies by to avoid unwrap errors
   :param m1: First mass (in M_sun)
   :param m2: Second mass (in M_sun)
   :param s1x: X component of first spin (dimensionless)
   :param s1y: Y component of first spin (dimensionless)
   :param s1z: Z component of first spin (dimensionless)
   :param s2x: X component of second spin (dimensionless)
   :param s2y: Y component of second spin (dimensionless)
   :param s2z: Z component of second spin (dimensionless)
   :param l1: Tidal deformalbility of first mass
   :param l2: Tidal deformalbility of second mass
   :param approximant: LAL string for approximant
   :param unwrap: Flag indicating whether to unwrap and return amp and phase
   :param phase:
       If known, pass phase to use, else pass "random" to indicate
       a random phase
   :param deltaF: If on a regular grid, frequency spacing (Hz)
   :param f_min:
       If on a regular grid, minimum frequency to query the approximant at (Hz)
   :param f_max:
       If on a regular grid, maximum frequency (Hz, ~Nyquist frequency)
   :return:
       1. len(fs_gen / sub_fac) array with amplitude of frequency domain
          waveform (units of 1/Hz)
       2. len(fs_gen / sub_fac) array with phase of frequency domain waveform
