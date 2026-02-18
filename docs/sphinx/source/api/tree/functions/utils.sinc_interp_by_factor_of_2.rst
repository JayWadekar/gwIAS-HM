utils.sinc_interp_by_factor_of_2
================================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def sinc_interp_by_factor_of_2(t, x, left_ind = None, right_ind = None, support = params.SUPPORT_SINC_FILTER, n_interp = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``t``
     - -
     - -
     - Time axis of the original x array.
   * - ``x``
     - -
     - -
     - Samples to be interpolated. The last axis of the array will be interpolated.
   * - ``left_ind``
     - -
     - None
     - If known, left index of region of interest
   * - ``right_ind``
     - -
     - None
     - If known, right index of region of interest
   * - ``support``
     - -
     - params.SUPPORT_SINC_FILTER
     - Support of the interpolation filter
   * - ``n_interp``
     - -
     - 1
     - Number of times to sinc-interpolate by factor of 2

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - t_interp, x_interp # Apologies to grad students Length of x_interp is 2\\\*len(x) - 4\\\*(support+1) if n_interp = 1

Docstring
---------

.. code-block:: text

   :param t: Time axis of the original x array.
   :param x: Samples to be interpolated. The last axis of the 
             array will be interpolated.
   :param left_ind: If known, left index of region of interest
   :param right_ind: If known, right index of region of interest
   :param support: Support of the interpolation filter
   :param n_interp: Number of times to sinc-interpolate by factor of 2
   :return: t_interp, x_interp
   # Apologies to grad students
   Length of x_interp is 2*len(x) - 4*(support+1) if n_interp = 1
