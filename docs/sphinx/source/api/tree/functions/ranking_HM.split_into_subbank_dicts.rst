ranking_HM.split_into_subbank_dicts
===================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def split_into_subbank_dicts(event_list, loc_id_index, skip_index = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``event_list``
     - -
     - -
     - One of the lists in bg_fg_max/cands_preveto_max/cands_postveto_max
   * - ``loc_id_index``
     - -
     - -
     - Index of the loc_id in each element of the list
   * - ``skip_index``
     - -
     - True
     - Flag to skip the loc_id in the resulting dictionary entry

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Dict with keys = loc_ids, and values = list of lists with each list containing a property of the events other than loc_id. This is like an entry of bg_fg_by_subbank, but with the values being lists of lists instead of lists of numpy arrays

Docstring
---------

.. code-block:: text

   :param event_list:
       One of the lists in bg_fg_max/cands_preveto_max/cands_postveto_max
   :param loc_id_index: Index of the loc_id in each element of the list
   :param skip_index: Flag to skip the loc_id in the resulting dictionary entry
   :return:
       Dict with keys = loc_ids, and values = list of lists with each list
       containing a property of the events other than loc_id.
       This is like an entry of bg_fg_by_subbank, but with the values being
       lists of lists instead of lists of numpy arrays
