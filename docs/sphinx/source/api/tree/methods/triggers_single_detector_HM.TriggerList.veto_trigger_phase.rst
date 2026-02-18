triggers_single_detector_HM.TriggerList.veto_trigger_phase
==========================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Chi2 and tail veto using phase

Signature
---------

.. code-block:: python

   def veto_trigger_phase(self, trigger, dcalphas = None, n_chunk = params.N_CHUNK, cov_degrade = params.COV_DEGRADE, chi2_min_eigval = params.CHI2_MIN_EIGVAL, pthresh_chi2 = params.THRESHOLD_CHI2, split_chunks = params.SPLIT_CHUNKS, pthresh_split = params.THRESHOLD_SPLIT, subset_details = None, bestfit_wf = None, test_injection = False, physical_mode_ratio = False, verbose = True, lazy = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - -
     - Trigger that jumped (suggest using optimized trigger, but should be OK otherwise)
   * - ``dcalphas``
     - -
     - None
     - Array with spacings in calphas to allow mismatch in
   * - ``n_chunk``
     - -
     - params.N_CHUNK
     - Number of chunks to split the whitened waveform into
   * - ``cov_degrade``
     - -
     - params.COV_DEGRADE
     - Keep eigenvectors with eigenvalue > (max eigenvalue of covariance matrix \\\* cov_degrade)
   * - ``chi2_min_eigval``
     - -
     - params.CHI2_MIN_EIGVAL
     - If the highest eigenvalue of the covariance matrix is below this, do not perform chi-squared test
   * - ``pthresh_chi2``
     - -
     - params.THRESHOLD_CHI2
     - Probability of the overall chi2 veto firing due to Gaussian noise
   * - ``split_chunks``
     - -
     - params.SPLIT_CHUNKS
     - List of pairs of lists of chunk indices (each entry of the form [[ind11, ind12, ...], [ind21, ind22, ...]]) with the members of each pair a set of chunks to contrast with the others)
   * - ``pthresh_split``
     - -
     - params.THRESHOLD_SPLIT
     - Probability of each split veto firing due to Gaussian noise
   * - ``subset_details``
     - -
     - None
     - If we have already defined an appropriate subset, 3-tuple as in the output of prepare_subset_for_vetoes
   * - ``bestfit_wf``
     - -
     - None
     - If known, pass the bestfit waveform (we can use any other waveform, as long as we take care of the indices). Pass 3 modes
   * - ``test_injection``
     - -
     - False
     - Use the injected waveform as the bestfit waveform
   * - ``physical_mode_ratio``
     - -
     - False
     - require A_33/A_22 and A_44/A_22 of the best-fit wf to be in physically allowed region
   * - ``verbose``
     - -
     - True
     - Flag indicating whether to print warnings
   * - ``lazy``
     - -
     - True
     - Flag indicating whether to stop after a test fails

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 0. True/False if the trigger passes/fails the veto 1. Boolean array of size 2 + len(split_chunks) with zeros marking glitch tests that fired The indices correspond to: 0: No chunks present 1: Overall chi^2 test 2: 2 + len(split_chunks): Split tests

Docstring
---------

.. code-block:: text

   Chi2 and tail veto using phase
   :param trigger: Trigger that jumped (suggest using optimized trigger,
                   but should be OK otherwise)
   :param dcalphas: Array with spacings in calphas to allow mismatch in
   :param n_chunk: Number of chunks to split the whitened waveform into
   :param cov_degrade:
       Keep eigenvectors with eigenvalue > (max eigenvalue of covariance
       matrix * cov_degrade)
   :param chi2_min_eigval:
       If the highest eigenvalue of the covariance matrix is below this,
       do not perform chi-squared test
   :param pthresh_chi2:
       Probability of the overall chi2 veto firing due to Gaussian noise
   :param split_chunks:
       List of pairs of lists of chunk indices (each entry of the form
       [[ind11, ind12, ...], [ind21, ind22, ...]]) with the members of
       each pair a set of chunks to contrast with the others)
   :param pthresh_split:
       Probability of each split veto firing due to Gaussian noise
   :param subset_details:
       If we have already defined an appropriate subset, 3-tuple as in
       the output of prepare_subset_for_vetoes
   :param bestfit_wf:
       If known, pass the bestfit waveform (we can use any other waveform,
       as long as we take care of the indices). Pass 3 modes
   :param physical_mode_ratio: require A_33/A_22 and A_44/A_22 of the best-fit wf
                               to be in physically allowed region
   :param test_injection: Use the injected waveform as the bestfit waveform
   :param verbose: Flag indicating whether to print warnings
   :param lazy: Flag indicating whether to stop after a test fails
   :return: 0. True/False if the trigger passes/fails the veto
            1. Boolean array of size 2 + len(split_chunks) with zeros
               marking glitch tests that fired
               The indices correspond to:
               0: No chunks present
               1: Overall chi^2 test
               2: 2 + len(split_chunks): Split tests
