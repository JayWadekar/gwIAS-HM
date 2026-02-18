coincidence_HM.collect_all_candidate_files_npz
==============================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def collect_all_candidate_files_npz(dir_path, collect_before_veto = True, collect_rerun = True, collect_timeseries = False, detectors = ('H1', 'L1'), ncores = 1)

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
   * - ``collect_before_veto``
     - -
     - True
     - Flag whether to return results before vetoes. If False, we will only return the vetoed candidates
   * - ``collect_rerun``
     - -
     - True
     - Flag indicating whether to collect rerun files
   * - ``collect_timeseries``
     - -
     - False
     - Flag whether to collect timeseries
   * - ``detectors``
     - -
     - ('H1', 'L1')
     - Tuple with names of the two detectors we will be loading results for
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
     - events, veto_metadata, coherent_scores, veto_metadata_keys (+ timeseries if collect_timeseries)

Docstring
---------

.. code-block:: text

   :param dir_path:
   :param collect_before_veto:
       Flag whether to return results before vetoes. If False, we will only
       return the vetoed candidates
   :param collect_rerun: Flag indicating whether to collect rerun files
   :param collect_timeseries: Flag whether to collect timeseries
   :param detectors:
       Tuple with names of the two detectors we will be loading results for
   :param ncores: Number of cores to use for collection
   :return:
       events, veto_metadata, coherent_scores, veto_metadata_keys
       (+ timeseries if collect_timeseries)
