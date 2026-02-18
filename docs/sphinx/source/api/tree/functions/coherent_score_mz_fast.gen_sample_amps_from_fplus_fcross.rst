coherent_score_mz_fast.gen_sample_amps_from_fplus_fcross
========================================================

Back to :doc:`Module page <../modules/coherent_score_mz_fast>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def gen_sample_amps_from_fplus_fcross(fplus, fcross, mu, psi)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fplus``
     - -
     - -
     - Response to the plus polarization for psi = 0
   * - ``fcross``
     - -
     - -
     - Response to the cross polarization for psi = 0
   * - ``mu``
     - -
     - -
     - Inclination
   * - ``psi``
     - -
     - -
     - Polarization angle

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

   :param fplus: Response to the plus polarization for psi = 0
   :param fcross: Response to the cross polarization for psi = 0
   :param mu: Inclination
   :param psi: Polarization angle
   :returns A_p + 1j * A_c
   ## Note that this seems to have the wrong convention for mu
