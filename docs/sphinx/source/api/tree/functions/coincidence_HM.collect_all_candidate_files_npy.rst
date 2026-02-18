coincidence_HM.collect_all_candidate_files_npy
==============================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def collect_all_candidate_files_npy(dir_path, collect_after_veto = True, collect_before_veto = False, collect_rerun = False, collect_timeseries = False, ncores = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dir_path``
     - -
     - -
     - -
   * - ``collect_after_veto``
     - -
     - True
     - -
   * - ``collect_before_veto``
     - -
     - False
     - -
   * - ``collect_rerun``
     - -
     - False
     - Flag indicating whether to collect rerun files
   * - ``collect_timeseries``
     - -
     - False
     - Flag whether to collect timeseries (assumes we collected both before/after in the run)
   * - ``ncores``
     - -
     - 1
     - Number of cores to use for collection

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

   :param dir_path:
   :param collect_after_veto:
   :param collect_before_veto:
   :param collect_rerun: Flag indicating whether to collect rerun files
   :param collect_timeseries:
       Flag whether to collect timeseries (assumes we collected both
       before/after in the run)
   :param ncores: Number of cores to use for collection
   :return:
