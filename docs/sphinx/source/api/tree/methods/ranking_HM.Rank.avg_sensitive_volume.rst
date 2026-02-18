ranking_HM.Rank.avg_sensitive_volume
====================================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

Run after scoring the bank bg

Signature
---------

.. code-block:: python

   def avg_sensitive_volume(self, subbank_id)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``subbank_id``
     - -
     - -
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
     - Avg sensitive volume in relative units The reference point is at 1/1 normfac ratio and self.median_normfacs_by_subbank[subbank_id][0]

Docstring
---------

.. code-block:: text

   Run after scoring the bank bg
   :return: Avg sensitive volume in relative units
   The reference point is at 1/1 normfac ratio and
   self.median_normfacs_by_subbank[subbank_id][0]
