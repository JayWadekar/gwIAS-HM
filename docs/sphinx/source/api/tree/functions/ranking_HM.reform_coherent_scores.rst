ranking_HM.reform_coherent_scores
=================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Edits input structures in place via aliasing and returns scores_bg_by_subbank

Signature
---------

.. code-block:: python

   def reform_coherent_scores(cands_preveto_max, median_normfacs_by_subbank)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``cands_preveto_max``
     - -
     - -
     - List (bg/fg/lvc/inj) of lists (n_trigger) with each element having a set of event attributes (prior terms, pclists, loc_id, sensitivity params, other stuff)
   * - ``median_normfacs_by_subbank``
     - -
     - -
     - Dictionary with n_det array of median normfacs for each subbank ID

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Recomputes the median, and edits the prior terms and median normfacs in place. It also returns the background split by subbank

Docstring
---------

.. code-block:: text

   Edits input structures in place via aliasing and returns
   scores_bg_by_subbank
   :param cands_preveto_max:
       List (bg/fg/lvc/inj) of lists (n_trigger) with each element having a set
       of event attributes
       (prior terms, pclists, loc_id, sensitivity params, other stuff)
   :param median_normfacs_by_subbank:
       Dictionary with n_det array of median normfacs for each subbank ID
   :return:
       Recomputes the median, and edits the prior terms and median normfacs in
       place. It also returns the background split by subbank
