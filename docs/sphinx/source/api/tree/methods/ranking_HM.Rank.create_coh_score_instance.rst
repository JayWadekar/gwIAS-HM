ranking_HM.Rank.create_coh_score_instance
=========================================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

Removed from init to make saved rank objects independent of the cogwheel version

Signature
---------

.. code-block:: python

   def create_coh_score_instance(self, cs_ver = 'JR', cs_table = None, example_trigs = None, **cs_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``cs_ver``
     - -
     - 'JR'
     - Version of coherent score to use, can be 'JR', 'O2', 'mz'
   * - ``cs_table``
     - -
     - None
     - Path to file with coherent score table (can be dictionary mapping (bank_id, subbank ID) to file), in case cs_ver == 'mz', it is instead a path to the appropriate coherent score npz file
   * - ``example_trigs``
     - -
     - None
     - List of example triggers to use for coherent score
   * - ``\*\*cs_kwargs``
     - -
     - -
     - Additional arguments to pass to the coherent score

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Instance of coherent score

Docstring
---------

.. code-block:: text

   Removed from init to make saved rank objects independent of
   the cogwheel version
   :param cs_ver:
       Version of coherent score to use, can be 'JR', 'O2', 'mz'
   :param cs_table:
       Path to file with coherent score table (can be dictionary mapping
       (bank_id, subbank ID) to file), in case cs_ver == 'mz', it is
       instead a path to the appropriate coherent score npz file
   :param example_trigs: List of example triggers to use for coherent score
   :param cs_kwargs: Additional arguments to pass to the coherent score
   :return: Instance of coherent score
