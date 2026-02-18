template_bank_generator_HM.gen_random_pars
==========================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Generates n_wf random parameters. Applies either mrng (preferably) or mtrng, along with qrng

Signature
---------

.. code-block:: python

   def gen_random_pars(n_wf, mtrng, qrng, mrng, srng, lrng, flat_in_chieff = False, mcrng = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``n_wf``
     - -
     - -
     - Number of random parameter choices
   * - ``mtrng``
     - -
     - -
     - Array of length 2 with range of total masses (in M_sun)
   * - ``qrng``
     - -
     - -
     - Array of length 2 with range of mass ratios (q<1)
   * - ``mrng``
     - -
     - -
     - Array/tuple of length 2 with range of component mass (in M_sun)
   * - ``srng``
     - -
     - -
     - Array of length 2 with range of spins
   * - ``lrng``
     - -
     - -
     - Array of length 2 with range of tidal deformabilities
   * - ``flat_in_chieff``
     - -
     - False
     - Bool, if to take s1z,s2z uniform from srng (False) or take chi_eff = m1\\\*s1z+m2\\\*s2z/(m1+m2) uniform in snrg, and take chia = s1z-s2z uniform in the allowed range given chieff (True)
   * - ``mcrng``
     - -
     - None
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

   Generates n_wf random parameters. Applies either mrng (preferably) or
   mtrng, along with qrng
   :param n_wf: Number of random parameter choices
   :param mtrng: Array of length 2 with range of total masses (in M_sun)
   :param qrng: Array of length 2 with range of mass ratios (q<1)
   :param mrng: Array/tuple of length 2 with range of component mass (in M_sun)
   :param srng: Array of length 2 with range of spins
   :param lrng: Array of length 2 with range of tidal deformabilities
   :param flat_in_chieff : Bool, if to take s1z,s2z uniform from srng (False) or take
                           chi_eff = m1*s1z+m2*s2z/(m1+m2) uniform in snrg,
                           and take chia = s1z-s2z uniform in the allowed range
                           given chieff (True)
