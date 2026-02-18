template_bank_generator_HM.TemplateBank.grid_range
==================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Returns a grid between min_val and max_val If force_zero=True: 0 is a gridpoint Spacing is <= d_val/2 at the edges Spacing is <= d_val in the bulk If force_zero=False: Spacing is <= d_val/2 at the edges Spacing is exactly d_val in the bulk

Signature
---------

.. code-block:: python

   def grid_range(min_val, max_val, d_val, force_zero = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``min_val``
     - -
     - -
     - -
   * - ``max_val``
     - -
     - -
     - -
   * - ``d_val``
     - -
     - -
     - -
   * - ``force_zero``
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

   Returns a grid between min_val and max_val
   If force_zero=True:
       0 is a gridpoint
       Spacing is <= d_val/2 at the edges
       Spacing is <= d_val in the bulk
   If force_zero=False:
       Spacing is <= d_val/2 at the edges
       Spacing is exactly d_val in the bulk
