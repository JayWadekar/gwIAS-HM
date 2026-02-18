template_bank_generator_HM.TemplateBank.closest_coeff
=====================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Return the values of the coefficients in coeff_grid that most closely match some input coefficients. Similar to rounding to the grid spacing, but accounts for offset and the grid needn't be uniform or rectangular, e.g. if nonphysical templates have been removed.

Signature
---------

.. code-block:: python

   def closest_coeff(coeffs, coeff_grid)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``coeffs``
     - -
     - -
     - n_wfs x n_dims array with the coefficients of a number of waveforms
   * - ``coeff_grid``
     - -
     - -
     - n_templates x n_dims array with the coefficients of the templates in the bank. Doesn't need to be a regular grid

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

   Return the values of the coefficients in coeff_grid that most
   closely match some input coefficients. Similar to rounding to the
   grid spacing, but accounts for offset and the grid needn't be uniform
   or rectangular, e.g. if nonphysical templates have been removed.
   :param coeffs:
       n_wfs x n_dims array with the coefficients of a number of waveforms
   :param coeff_grid:
       n_templates x n_dims array with the coefficients of the templates in
       the bank. Doesn't need to be a regular grid
