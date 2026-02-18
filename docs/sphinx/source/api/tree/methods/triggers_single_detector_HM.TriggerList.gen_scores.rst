triggers_single_detector_HM.TriggerList.gen_scores
==================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Convenience function to generate overlaps and hole corrections, for multiple waveforms or calphas, on full data or a subset of it

Signature
---------

.. code-block:: python

   def gen_scores(self, wfs_whitened_fd = None, calphas = None, subset = False, zero_invalid = True, only_22 = False, orthogonalize_modes = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wfs_whitened_fd``
     - -
     - None
     - nwf x 3 x length rfft(fftsize) complex array with frequency domain waveforms Convention: power is towards the right side
   * - ``calphas``
     - -
     - None
     - Optional nwf x n_pars array with coefficients of basis functions, overrides waveforms if given, (can be vector for nwf = 1)
   * - ``subset``
     - -
     - False
     - Flag indicating whether to generate triggers on full data or subset
   * - ``zero_invalid``
     - -
     - True
     - Flag indicating whether to zero invalid scores
   * - ``only_22``
     - -
     - False
     - Use only 22 instead of the 3 modes everywhere
   * - ``orthogonalize_modes``
     - -
     - True
     - Orthogonalizes the different mode wfs If false, returns the covariance matrix between modes (upper triangular elements)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. nwf x 3 x len(data/data_sub) array with overlaps (zero where the hole correction cannot be trusted) 2. nwf x 3 x len(data/data_sub) array with hole corrections 3. nwf x 3 x len(data/data_sub) Boolean array indicating validity of overlaps Can be vectors if wfs_whitened_fd or calphas is a vector

Docstring
---------

.. code-block:: text

   Convenience function to generate overlaps and hole corrections, for
   multiple waveforms or calphas, on full data or a subset of it
   :param wfs_whitened_fd:
       nwf x 3 x length rfft(fftsize) complex array with frequency domain
       waveforms
       Convention: power is towards the right side
   :param calphas:
       Optional nwf x n_pars array with coefficients of basis functions,
       overrides waveforms if given, (can be vector for nwf = 1)
   :param subset:
       Flag indicating whether to generate triggers on full data or subset
   :param zero_invalid: Flag indicating whether to zero invalid scores
   :param only_22: Use only 22 instead of the 3 modes everywhere
   :param orthogonalize_modes: Orthogonalizes the different mode wfs
       If false, returns the covariance matrix between modes 
       (upper triangular elements)
   :return:
       1. nwf x 3 x len(data/data_sub) array with overlaps
          (zero where the hole correction cannot be trusted)
       2. nwf x 3 x len(data/data_sub) array with hole corrections
       3. nwf x 3 x len(data/data_sub) Boolean array indicating validity of
          overlaps
       Can be vectors if wfs_whitened_fd or calphas is a vector
