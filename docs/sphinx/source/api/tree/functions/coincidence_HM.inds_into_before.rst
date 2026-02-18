coincidence_HM.inds_into_before
===============================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Gives indices into cand_before_veto for triggers in cand_after_veto We can use tolist(), but this is almost 10 times faster

Signature
---------

.. code-block:: python

   def inds_into_before(cand_after_veto, cand_before_veto, isort_after, isort_before)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``cand_after_veto``
     - -
     - -
     - n1 x n_det x (row of processedclist) array with candidates after vetoes
   * - ``cand_before_veto``
     - -
     - -
     - (n2 > n1) x n_det x (row of processedclist) array with candidates before vetoes
   * - ``isort_after``
     - -
     - -
     - n1 array with arrays for lexicographic sort of cand_after_veto
   * - ``isort_before``
     - -
     - -
     - n2 array with arrays for lexicographic sort of cand_before_veto

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

   Gives indices into cand_before_veto for triggers in cand_after_veto
   We can use tolist(), but this is almost 10 times faster
   :param cand_after_veto:
       n1 x n_det x (row of processedclist) array with candidates
       after vetoes
   :param cand_before_veto:
       (n2 > n1) x n_det x (row of processedclist) array with candidates
       before vetoes
   :param isort_after:
       n1 array with arrays for lexicographic sort of cand_after_veto
   :param isort_before:
       n2 array with arrays for lexicographic sort of cand_before_veto
   :return:
