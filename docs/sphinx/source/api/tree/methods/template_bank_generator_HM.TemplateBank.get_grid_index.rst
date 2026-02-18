template_bank_generator_HM.TemplateBank.get_grid_index
======================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Get the multiindex of the gridpoint that most closely matches a point. Assumes that the grid is rectangular, but not necessarily regular.

Signature
---------

.. code-block:: python

   def get_grid_index(grid_axes, point)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``grid_axes``
     - -
     - -
     - list of arrays with the coordinates of the gridpoints, one for each grid dimension.
   * - ``point``
     - -
     - -
     - array with the coordinates of the point we want to match

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

   Get the multiindex of the gridpoint that most closely
   matches a point. Assumes that the grid is rectangular, but not
   necessarily regular.
   :param grid_axes: list of arrays with the coordinates of the
                     gridpoints, one for each grid dimension.
   :param point: array with the coordinates of the point we want to match
