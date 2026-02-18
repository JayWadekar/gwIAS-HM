ranking_HM.veto_subthreshold_top_cands
======================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Top candidates in the cands_preveto_max list which are subthreshold (i.e. have SNR below min_veto_chi2) are passed through all the veto tests again

Signature
---------

.. code-block:: python

   def veto_subthreshold_top_cands(rank_obj, ncores = 1, num_redo_bkg = 5000, triggers_dir = None, rerun_coh_score = True, fname_to_save = None, score_final_object = True, **ranking_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``rank_obj``
     - -
     - -
     - Individual Rank object (not a list)
   * - ``ncores``
     - -
     - 1
     - -
   * - ``num_redo_bkg``
     - -
     - 5000
     - How many background triggers (ranked by IFAR starting from the highest) to redo vetos for
   * - ``triggers_dir``
     - -
     - None
     - Directory where the triggers are stored e.g., '/data/jayw/IAS/GW/Data/HM_O3b_search/OutputDir/'
   * - ``rerun_coh_score``
     - -
     - True
     - Flag whether to rerun the coherent score for all the top candidates (not just the subthreshold ones). The reason to do this is that coherent score is now calculated at a higher resolution compared the version in coincidence_HM.py (the coh score is now averaged over 10 iterations and calculated with slightly larger log2n_qmc and nphi)
   * - ``fname_to_save``
     - -
     - None
     - If known, filename to save the Rank object with the new vetoes to. If None, we apply the vetoes in place, but don't save the object
   * - ``score_final_object``
     - -
     - True
     - Flag whether to score the final object
   * - ``\*\*ranking_kwargs``
     - -
     - -
     - Any extra arguments to pass to the ranking function. Used only if score_final_object is True Below is an example script for running this code on a cluster: salloc --nodes=1 --cpus-per-task=24 --time=12:00:00 --mem-per-cpu=4GB python import sys sys.path.insert(0,".../gw_detection_ias") import ranking_HM as rank # Need to load in the mode r+ as we will modify veto metadatas of triggers rank_obj = rank.Rank.from_hdf5('--.hdf5', mode="r+") rank.veto_subthreshold_top_cands(rank_obj, ncores=24, num_redo_bkg=5000, triggers_dir='--', fname_to_save='---.hdf5') # Safely close the file rank_obj.fobj.close()

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

   Top candidates in the cands_preveto_max list which are subthreshold
   (i.e. have SNR below min_veto_chi2)
   are passed through all the veto tests again
   
   :param rank_obj: Individual Rank object (not a list)
   :param ncores:
   :param num_redo_bkg:
       How many background triggers (ranked by IFAR starting from the highest)
       to redo vetos for
   :param triggers_dir: Directory where the triggers are stored
       e.g., '/data/jayw/IAS/GW/Data/HM_O3b_search/OutputDir/'
   :param rerun_coh_score:
       Flag whether to rerun the coherent score for all the top candidates
       (not just the subthreshold ones).
       The reason to do this is that coherent score is now calculated at a
       higher resolution compared the version in coincidence_HM.py
       (the coh score is now averaged over 10 iterations and calculated with
       slightly larger log2n_qmc and nphi)
   :param fname_to_save:
       If known, filename to save the Rank object with the new vetoes to.
       If None, we apply the vetoes in place, but don't save the object
   :param score_final_object: Flag whether to score the final object
   :param ranking_kwargs:
       Any extra arguments to pass to the ranking function. Used only if
       score_final_object is True
   
   Below is an example script for running this code on a cluster:
       salloc --nodes=1 --cpus-per-task=24 --time=12:00:00 --mem-per-cpu=4GB
       python
       import sys
       sys.path.insert(0,".../gw_detection_ias")
       import ranking_HM as rank
       # Need to load in the mode r+ as we will modify veto metadatas of triggers
       rank_obj = rank.Rank.from_hdf5('--.hdf5', mode="r+")
       rank.veto_subthreshold_top_cands(rank_obj, ncores=24, num_redo_bkg=5000,
           triggers_dir='--', fname_to_save='---.hdf5')
       # Safely close the file
       rank_obj.fobj.close()
