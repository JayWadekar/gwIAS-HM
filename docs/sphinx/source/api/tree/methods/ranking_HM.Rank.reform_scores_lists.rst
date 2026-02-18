ranking_HM.Rank.reform_scores_lists
===================================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

After maximizing over banks, there are fewer entries in cands_preveto_max, which are no longer in order of subbank_id. Apart from the return, this function 1. Recomputes the median normfacs for the subbanks 2. Reevaluates the sensitivity penalties for all candidates 3. Reorders the entries in cands_preveto_max

Signature
---------

.. code-block:: python

   def reform_scores_lists(self)

This callable has no explicit input variables.

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - A list of len(nsubbanks) with scores_bg_by_subbank_nonvetoed

Docstring
---------

.. code-block:: text

   After maximizing over banks, there are fewer entries in
   cands_preveto_max, which are no longer in order of subbank_id. Apart
   from the return, this function
   1. Recomputes the median normfacs for the subbanks
   2. Reevaluates the sensitivity penalties for all candidates
   3. Reorders the entries in cands_preveto_max
   :return: A list of len(nsubbanks) with scores_bg_by_subbank_nonvetoed
