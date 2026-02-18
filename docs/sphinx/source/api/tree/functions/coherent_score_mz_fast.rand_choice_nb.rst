coherent_score_mz_fast.rand_choice_nb
=====================================

Back to :doc:`Module page <../modules/coherent_score_mz_fast>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def rand_choice_nb(arr, cprob, nvals)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``arr``
     - -
     - -
     - A nD numpy array of values to sample from
   * - ``cprob``
     - -
     - -
     - A 1D numpy array of cumulative probabilities for the given samples
   * - ``nvals``
     - -
     - -
     - Number of samples desired

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - nvals random samples from the given array with a given probability

Docstring
---------

.. code-block:: text

   :param arr: A nD numpy array of values to sample from
   :param cprob:
       A 1D numpy array of cumulative probabilities for the given samples
   :param nvals: Number of samples desired
   :return: nvals random samples from the given array with a given probability
