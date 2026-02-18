template_bank_generator_HM.TemplateBank.get_approximate_params
==============================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Return a guess of m1, m2, s1z, s2z, l1, l2 from a template's coefficients We have currently only used 22, not HM (for which, breaking the q-Chieff degeneracy might help)

Signature
---------

.. code-block:: python

   def get_approximate_params(self, calpha)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``calpha``
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

   Return a guess of m1, m2, s1z, s2z, l1, l2 from
   a template's coefficients
   We have currently only used 22, not HM (for which, breaking the q-Chieff
    degeneracy might help)
