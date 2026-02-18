ranking_HM.maximization_info_from_pclists
=========================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Returns information useful for maximize_over_groups

Signature
---------

.. code-block:: python

   def maximization_info_from_pclists(pclists, incoherent_score_func = utils.incoherent_score)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``pclists``
     - -
     - -
     - n_candidates x n_det x len(processedclist) array
   * - ``incoherent_score_func``
     - -
     - utils.incoherent_score
     - Function that accepts processedclists for trigger(s) and returns incoherent score(s) used for ranking

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_candidates x ... array with t_H, t_L, .., incoherent score used for ranking, rho^2

Docstring
---------

.. code-block:: text

   Returns information useful for maximize_over_groups
   :param pclists: n_candidates x n_det x len(processedclist) array
   :param incoherent_score_func:
       Function that accepts processedclists for trigger(s) and returns
       incoherent score(s) used for ranking
   :return:
       n_candidates x ... array with
       t_H, t_L, .., incoherent score used for ranking, rho^2
