utils.sinc_interp_x2D
=====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

exact same as sinc_interp_by_factor_of_2() except along last axis of two dimensional array x2D

Signature
---------

.. code-block:: python

   def sinc_interp_x2D(t, x2D, left_ind = None, right_ind = None, support = params.SUPPORT_SINC_FILTER, n_interp = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``t``
     - -
     - -
     - -
   * - ``x2D``
     - -
     - -
     - -
   * - ``left_ind``
     - -
     - None
     - -
   * - ``right_ind``
     - -
     - None
     - -
   * - ``support``
     - -
     - params.SUPPORT_SINC_FILTER
     - -
   * - ``n_interp``
     - -
     - 1
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

   exact same as sinc_interp_by_factor_of_2() except along last axis of
   two dimensional array x2D
