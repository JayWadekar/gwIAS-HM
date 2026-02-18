template_bank_generator_HM.TemplateBank.marginalized_HM_scores
==============================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

The scores obtained from triggering can sometimes be unphysical ,e.g., \\\|Z_33\\\|>>\\\|Z_22\\\| (here Z=complex rho timeseries) This function marginalizes or maximizes over inclination and mass ratio for HMs

Signature
---------

.. code-block:: python

   def marginalized_HM_scores(self, triggers, input_Z = False, marginalized = True, Rij_samples = None, single_det = True, calpha = None, N_det_effective = 2, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``triggers``
     - -
     - -
     - if input_Z: array with ntriggers x [Z22, Z33, Z44] else: if single_det: array with ntriggers x processedclist else: array with n_det x processedclist (only implemented for n_triggers=1)
   * - ``input_Z``
     - -
     - False
     - -
   * - ``marginalized``
     - -
     - True
     - marginalizing instead of maximizing
   * - ``Rij_samples``
     - -
     - None
     - -
   * - ``single_det``
     - -
     - True
     - Boolean flag for multi-detector case
   * - ``calpha``
     - -
     - None
     - [c0,c1,...] array (in case Norm Flow is implemented in the future)
   * - ``N_det_effective``
     - -
     - 2
     - if single_det=True, how many times to multiply the likelihood in single det case by corresponding to the likelihood expected from other detectors in an optimistic scenario.
   * - ``\*\*kwargs``
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
     - if marginalized: return Z^2_marginalized (size: ntriggers) else: ntriggers x [Z22, Z33, Z44] array where Z are complex overlaps corresponding to max lnL physical sample

Docstring
---------

.. code-block:: text

   The scores obtained from triggering can sometimes be unphysical
   ,e.g., |Z_33|>>|Z_22| (here Z=complex rho timeseries)
   This function marginalizes or maximizes over inclination and mass ratio for HMs
   :param triggers:
       if input_Z:
               array with ntriggers x [Z22, Z33, Z44]
       else:
           if single_det:
               array with ntriggers x processedclist
           else:
               array with n_det x processedclist (only implemented for n_triggers=1)
   :param marginalized: marginalizing instead of maximizing
   :param single_det: Boolean flag for multi-detector case
   :param calpha: [c0,c1,...] array (in case Norm Flow is implemented in the future)
   :param N_det_effective: if single_det=True, how many times to multiply the
           likelihood in single det case by  corresponding to the likelihood
           expected from other detectors in an optimistic scenario.
   :return: if marginalized: return Z^2_marginalized (size: ntriggers)
            else: ntriggers x [Z22, Z33, Z44] array
                    where Z are complex overlaps corresponding
                    to max lnL physical sample
