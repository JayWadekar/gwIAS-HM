ranking_HM.print_top_cand_list
==============================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Print the top candidates from the rank_objs

Signature
---------

.. code-block:: python

   def print_top_cand_list(rank_objs, allbanksbg_shifted, runs_per_yr, run, LVK = True, IFAR_THRESHOLD_SAVE = 0, IFAR_THRESHOLD_PRINT = 0)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``rank_objs``
     - -
     - -
     - List of rank objects
   * - ``allbanksbg_shifted``
     - -
     - -
     - Shifted background scores
   * - ``runs_per_yr``
     - -
     - -
     - Number of runs per year
   * - ``run``
     - -
     - -
     - Name of the run
   * - ``LVK``
     - -
     - True
     - Boolean flag for printing cands which do or do not overlap with known LVK events
   * - ``IFAR_THRESHOLD_SAVE``
     - -
     - 0
     - Threshold (/bank/year) for saving candidates in output dictionary
   * - ``IFAR_THRESHOLD_PRINT``
     - -
     - 0
     - Threshold (/bank/year) for printing candidates

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - nested dictionary with properties of the top candidates in each bank

Docstring
---------

.. code-block:: text

   Print the top candidates from the rank_objs
   :param rank_objs: List of rank objects
   :param allbanksbg_shifted: Shifted background scores
   :param runs_per_yr: Number of runs per year
   :param run: Name of the run
   :param LVK: Boolean flag for printing cands
       which do or do not overlap with known LVK events
   :param IFAR_THRESHOLD_SAVE: Threshold (/bank/year) for saving candidates
       in output dictionary
   :param IFAR_THRESHOLD_PRINT: Threshold (/bank/year) for printing candidates
   :return:
       nested dictionary with properties of the top candidates in each bank
