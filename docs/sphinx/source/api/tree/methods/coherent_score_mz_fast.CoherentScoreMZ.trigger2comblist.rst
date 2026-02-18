coherent_score_mz_fast.CoherentScoreMZ.trigger2comblist
=======================================================

Back to :doc:`Class page <../classes/coherent_score_mz_fast.CoherentScoreMZ>`

Summary
-------

Takes a trigger from plots_publication and produces input to the function for determining the coherent score, this can can inject, compute missing quantities, and make plots (assumes the trigger is from the class runtype)

Signature
---------

.. code-block:: python

   def trigger2comblist(self, trigger = None, timeseries = None, loc_id = None, ref_normfac = None, time_slide_jump = DEFAULT_TIMESLIDE_JUMP / 1000, dt = 0.1, lk_cut = 0.01, inject = (False, []), plot = False, ax = None, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - None
     - Array of size n_detector x row of processedclist
   * - ``timeseries``
     - -
     - None
     - If known, tuple of length n_detectors with n_samp x 3 array with Re(z), Im(z) (the output shares memory)
   * - ``loc_id``
     - -
     - None
     - Tuple with (bank_id, subbank_id), pass in if we're reading files
   * - ``ref_normfac``
     - -
     - None
     - Reference normfac if known, else we use the reference value from the saved run
   * - ``time_slide_jump``
     - -
     - DEFAULT_TIMESLIDE_JUMP / 1000
     - Least count of time slides (s)
   * - ``dt``
     - -
     - 0.1
     - length of time around trigger time to compute the overlaps
   * - ``lk_cut``
     - -
     - 0.01
     - Will only keep times when the likelihood is below the peak by this factor
   * - ``inject``
     - -
     - (False, [])
     - If desired, tuple with True/False, the list of injection parameters for each detector, and the calphas to use for MF
   * - ``plot``
     - -
     - False
     - True produces a plot for debugging purposes
   * - ``ax``
     - -
     - None
     - If known, axis to put the plot in
   * - ``\*\*kwargs``
     - -
     - -
     - Generic variable to capture any extra arguments

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Timeseries (if we had input, we pass it out) 2. Incoherent score of trigger 3. Shifts to apply to the timeseries > H1 to bring to zero lag 4. Array of length n_detector with detector sensitivities (useful when subtracting the incoherent piece)

Docstring
---------

.. code-block:: text

   Takes a trigger from plots_publication and produces input to the
   function for determining the coherent score, this can can inject,
   compute missing quantities, and make plots
   (assumes the trigger is from the class runtype)
   :param trigger: Array of size n_detector x row of processedclist
   :param timeseries: If known, tuple of length n_detectors with
       n_samp x 3 array with Re(z), Im(z) (the output shares memory)
   :param loc_id:
       Tuple with (bank_id, subbank_id), pass in if we're reading files
   :param ref_normfac:
       Reference normfac if known, else we use the reference value
       from the saved run
   :param time_slide_jump: Least count of time slides (s)
   :param dt: length of time around trigger time to compute the overlaps
   :param lk_cut: Will only keep times when the likelihood is below the
       peak by this factor
   :param inject:
       If desired, tuple with True/False, the list of injection
       parameters for each detector, and the calphas to use for MF
   :param plot: True produces a plot for debugging purposes
   :param ax: If known, axis to put the plot in
   :param kwargs: Generic variable to capture any extra arguments
   :return: 1. Timeseries (if we had input, we pass it out)
            2. Incoherent score of trigger
            3. Shifts to apply to the timeseries > H1 to bring to zero lag
            4. Array of length n_detector with detector sensitivities
               (useful when subtracting the incoherent piece)
