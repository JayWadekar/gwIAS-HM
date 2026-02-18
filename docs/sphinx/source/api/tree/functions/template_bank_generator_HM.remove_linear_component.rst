template_bank_generator_HM.remove_linear_component
==================================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Fits out a linear dependence of phase w.r.t frequency

Signature
---------

.. code-block:: python

   def remove_linear_component(phases, fs, wts, return_coeffs = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``phases``
     - -
     - -
     - n_wf x len(fs) array with angles (Note that it needs to be a 2d array)
   * - ``fs``
     - -
     - -
     - Array with frequencies
   * - ``wts``
     - -
     - -
     - Weights for linear fit, note that for Gaussian w/ variance sigma^2, weights are 1/sigma
   * - ``return_coeffs``
     - -
     - False
     - Flag to return coefficients of the linear terms

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. n_wf x len(fs) array with unweighted residuals of linear fit 2. If return_coeffs, n_wf x 2 array with coefficients of linear and constant terms (in that order)

Docstring
---------

.. code-block:: text

   Fits out a linear dependence of phase w.r.t frequency
   :param phases:
       n_wf x len(fs) array with angles (Note that it needs to be a 2d array)
   :param fs: Array with frequencies
   :param wts:
       Weights for linear fit, note that for Gaussian w/ variance sigma^2,
       weights are 1/sigma
   :param return_coeffs: Flag to return coefficients of the linear terms
   :return: 1. n_wf x len(fs) array with unweighted residuals of linear fit
            2. If return_coeffs, n_wf x 2 array with coefficients of linear
               and constant terms (in that order)
