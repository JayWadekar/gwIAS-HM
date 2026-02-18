template_bank_generator_HM.TemplateBank.make_grid_axes
======================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Return a list of arrays with the axes of a grid that covers all the physical waveforms used to generate the bank.

Signature
---------

.. code-block:: python

   def make_grid_axes(self, delta_calpha, fudge = params.TEMPLATE_SAFETY, force_zero = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``delta_calpha``
     - -
     - -
     - float, grid spacing
   * - ``fudge``
     - -
     - params.TEMPLATE_SAFETY
     - A safety factor to inflate the range of used values of each parameter
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

   Return a list of arrays with the axes of a grid that covers
   all the physical waveforms used to generate the bank.
   :param delta_calpha: float, grid spacing
   :param fudge: A safety factor to inflate the range of used values of
                 each parameter
