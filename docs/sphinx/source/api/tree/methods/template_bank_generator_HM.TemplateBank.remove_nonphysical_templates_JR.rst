template_bank_generator_HM.TemplateBank.remove_nonphysical_templates_JR
=======================================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

This function was written by Javier but now superseded Normalizing flow. Return the set of points from a rectangular grid in component space that are close to a physical waveform. Assumes that irrelevant c_alphas are zero (in the extra dimensions).

Signature
---------

.. code-block:: python

   def remove_nonphysical_templates_JR(self, grid_axes, fudge = params.TEMPLATE_SAFETY, ret_keep = False, delta_calpha = None, coeffs = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``grid_axes``
     - -
     - -
     - list of arrays that define the grid, ordered with decreasing number of components.
   * - ``fudge``
     - -
     - params.TEMPLATE_SAFETY
     - float, (fudge-1) is the size of the blob of grid-space to keep around each physical template, relative to the total extent of the grid in each dimension.
   * - ``ret_keep``
     - -
     - False
     - Flag indicating whether to return the boolean array indicating which gridpoints have been kept
   * - ``delta_calpha``
     - -
     - None
     - -
   * - ``coeffs``
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
     - array with the components of the templates in the grid that are close to some physical waveform

Docstring
---------

.. code-block:: text

   This function was written by Javier but now superseded
   Normalizing flow. 
   Return the set of points from a rectangular grid in component
   space that are close to a physical waveform.
   Assumes that irrelevant c_alphas are zero (in the extra dimensions).
   :param grid_axes: list of arrays that define the grid, ordered
                     with decreasing number of components.
   :param fudge: float, (fudge-1) is the size of the blob of
                           grid-space to keep around each physical template,
                           relative to the total extent of the grid
                           in each dimension.
   :param ret_keep: Flag indicating whether to return the boolean
                           array indicating which gridpoints have been kept
   :returns: array with the components of the templates in the grid
             that are close to some physical waveform
