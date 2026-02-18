template_bank_generator_HM.TemplateBank.get_coeff_grid
======================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Return the set of points on which to place templates, an array of dimension ntemplates x ndims

Signature
---------

.. code-block:: python

   def get_coeff_grid(self, delta_calpha = 0.7, fudge = params.TEMPLATE_SAFETY, remove_nonphysical = True, force_zero = True, ndims = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``delta_calpha``
     - -
     - 0.7
     - -
   * - ``fudge``
     - -
     - params.TEMPLATE_SAFETY
     - -
   * - ``remove_nonphysical``
     - -
     - True
     - -
   * - ``force_zero``
     - -
     - True
     - -
   * - ``ndims``
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

   Return the set of points on which to place templates, an array of
   dimension ntemplates x ndims
