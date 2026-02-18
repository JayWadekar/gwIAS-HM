utils.find_closest_coarse_calphas
=================================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Finds the closest associated coarse calphas to given fine calphas

Signature
---------

.. code-block:: python

   def find_closest_coarse_calphas(coarse_axes, fine_calphas)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``coarse_axes``
     - -
     - -
     - List of length n_axes, with the i^th entry being the array of allowed coarse calphas in dimension i
   * - ``fine_calphas``
     - -
     - -
     - Array with set of fine calphas whose associated coarse calphas we want, can be 1D for singleton

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_set x n_calpha array with coarse calphas assocated with each fine_calpha (always 2D)

Docstring
---------

.. code-block:: text

   Finds the closest associated coarse calphas to given fine calphas
   :param coarse_axes:
       List of length n_axes, with the i^th entry being the array of allowed
       coarse calphas in dimension i
   :param fine_calphas:
       Array with set of fine calphas whose associated coarse calphas we want,
       can be 1D for singleton
   :return:
       n_set x n_calpha array with coarse calphas assocated with each
       fine_calpha (always 2D)
