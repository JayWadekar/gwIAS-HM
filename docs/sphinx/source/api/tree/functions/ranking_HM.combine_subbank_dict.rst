ranking_HM.combine_subbank_dict
===============================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Populates a single list in bg_fg_max

Signature
---------

.. code-block:: python

   def combine_subbank_dict(dic_by_subbank)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dic_by_subbank``
     - -
     - -
     - Dict with keys = loc_ids, and values = list of arrays with each array containing a property of the events (an element of bg_fg_by_subbank)

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

   Populates a single list in bg_fg_max
   :param dic_by_subbank:
       Dict with keys = loc_ids, and values = list of arrays with each array
       containing a property of the events (an element of bg_fg_by_subbank)
   :return:
