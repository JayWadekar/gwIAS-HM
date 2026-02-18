template_bank_generator_HM.TemplateBank.marginalized_HM_scores_incl_temp_prior
==============================================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Adds the template prior to the marginalized HM scores Same arguments as marginalized_HM_scores() below

Signature
---------

.. code-block:: python

   def marginalized_HM_scores_incl_temp_prior(self, triggers, single_det = True, input_Z = False, marginalized = True, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``triggers``
     - -
     - -
     - if single_det: array with ntriggers x processedclist else: array with n_det x processedclist (only implemented for n_triggers=1)
   * - ``single_det``
     - -
     - True
     - -
   * - ``input_Z``
     - -
     - False
     - -
   * - ``marginalized``
     - -
     - True
     - -
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
     - -

Docstring
---------

.. code-block:: text

   Adds the template prior to the marginalized HM scores
   Same arguments as marginalized_HM_scores() below
   :param triggers:
       if single_det:
           array with ntriggers x processedclist
       else:
           array with n_det x processedclist (only implemented for n_triggers=1)
