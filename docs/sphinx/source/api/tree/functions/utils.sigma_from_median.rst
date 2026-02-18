utils.sigma_from_median
=======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Computes sigma from median for an array with Gaussian samps + outliers Silently returns 1 if we passed in an empty array

Signature
---------

.. code-block:: python

   def sigma_from_median(arr, axis = -1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``arr``
     - -
     - -
     - Array with samples
   * - ``axis``
     - -
     - -1
     - Axis to compute the sigma along

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

   Computes sigma from median for an array with Gaussian samps + outliers
   Silently returns 1 if we passed in an empty array
   :param arr: Array with samples
   :param axis: Axis to compute the sigma along
