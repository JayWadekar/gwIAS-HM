coincidence_HM.veto_and_optimize_group
======================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def veto_and_optimize_group(trigs_gp, trig_obj, time_shift_tol, veto_triggers = True, min_veto_chi2 = None, apply_threshold = True, relative_binning = True, origin = 0, opt_format = 'new')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigs_gp``
     - -
     - -
     - Processedclist of triggers
   * - ``trig_obj``
     - -
     - -
     - Instance of trig.TriggerList with processed data in the detector
   * - ``time_shift_tol``
     - -
     - -
     - -
   * - ``veto_triggers``
     - -
     - True
     - Flag indicating whether to veto triggers
   * - ``min_veto_chi2``
     - -
     - None
     - If given, veto only the candidates above this bar
   * - ``apply_threshold``
     - -
     - True
     - Flag to apply threshold on single-detector chi2 when optimizing
   * - ``relative_binning``
     - -
     - True
     - Flag to turn relative binning on/off
   * - ``origin``
     - -
     - 0
     - Origin in time (s) to split triggers relative to
   * - ``opt_format``
     - -
     - 'new'
     - How we choose the finer grid, changed between O1 and O2 analyses Exposed here to replicate old runs if needed

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Dictionary indexed by time / 0.01 s with 1. Pass/fail flag 2. Cloud of optimized triggers 3. Finer PSD drift correction factor if significant 4. Boolean array of size len(self.outlier_reasons) + 6 + len(split_chunks) with metadata 0: CBC_CAT2 flag ("and" of the values for the cloud) 1: CBC_CAT3 flag ("and" of the values for the cloud) The 2:len(self.outlier_reasons) + 6 + len(split_chunks) elements have zeros marking glitch tests that fired The indices correspond to: 2: len(self.outlier_reasons) + 2: index into outlier reasons for excess-power-like tests len(self.outlier_reasons) + 2: Finer PSD drift killed it len(self.outlier_reasons) + 3: No chunks present len(self.outlier_reasons) + 4: Overall chi-2 test len(self.outlier_reasons) + 5: len(self.outlier_reasons) + 5 + len(split_chunks): Split tests len(outlier_reasons) + 5 + len(split_chunks): Finer sinc-interpolation

Docstring
---------

.. code-block:: text

   :param trigs_gp: Processedclist of triggers
   :param trig_obj:
       Instance of trig.TriggerList with processed data in the detector
   :param time_shift_tol:
   :param veto_triggers: Flag indicating whether to veto triggers
   :param min_veto_chi2: If given, veto only the candidates above this bar
   :param apply_threshold:
       Flag to apply threshold on single-detector chi2 when optimizing
   :param relative_binning: Flag to turn relative binning on/off
   :param origin: Origin in time (s) to split triggers relative to
   :param opt_format:
       How we choose the finer grid, changed between O1 and O2 analyses
       Exposed here to replicate old runs if needed
   :return: Dictionary indexed by time / 0.01 s with
           1. Pass/fail flag
           2. Cloud of optimized triggers
           3. Finer PSD drift correction factor if significant
           4. Boolean array of size
               len(self.outlier_reasons) + 6 + len(split_chunks)
               with metadata
               0: CBC_CAT2 flag ("and" of the values for the cloud)
               1: CBC_CAT3 flag ("and" of the values for the cloud)
               The 2:len(self.outlier_reasons) + 6 + len(split_chunks) elements
               have zeros marking glitch tests that fired
               The indices correspond to:
               2: len(self.outlier_reasons) + 2: index into outlier reasons
                   for excess-power-like tests
               len(self.outlier_reasons) + 2: Finer PSD drift killed it
               len(self.outlier_reasons) + 3: No chunks present
               len(self.outlier_reasons) + 4: Overall chi-2 test
               len(self.outlier_reasons) + 5:
                   len(self.outlier_reasons) + 5 + len(split_chunks):
                       Split tests
               len(outlier_reasons) + 5 + len(split_chunks):
                   Finer sinc-interpolation
