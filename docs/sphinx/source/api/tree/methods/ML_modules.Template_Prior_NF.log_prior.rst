ML_modules.Template_Prior_NF.log_prior
======================================

Back to :doc:`Class page <../classes/ML_modules.Template_Prior_NF>`

Summary
-------

Compute the prior of the calpha templates.

Signature
---------

.. code-block:: python

   def log_prior(self, calpha_array)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``calpha_array``
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
     - Log of the prior density at the location of the template. For the probability of a template, use: np.exp(prior) \\\* (delta_calpha\\\*\\\*grid_ndims)

Docstring
---------

.. code-block:: text

   Compute the prior of the calpha templates.
   :param calpha_norm_array: array of calpha templates normalized
                             using bank.calpha_transformer.transform(calphas)
   :return: Log of the prior density at the location of the template.
       For the probability of a template, use:
       np.exp(prior) * (delta_calpha**grid_ndims)
