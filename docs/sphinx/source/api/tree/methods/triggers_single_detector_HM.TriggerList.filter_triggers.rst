triggers_single_detector_HM.TriggerList.filter_triggers
=======================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Apply cuts on triggers in self.filteredclist. Call with filters=None, rejects=None, and reset_filters=True to clear filters

Signature
---------

.. code-block:: python

   def filter_triggers(self, filters = None, rejects = None, reset_filters = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``filters``
     - -
     - None
     - -
   * - ``rejects``
     - -
     - None
     - -
   * - ``reset_filters``
     - -
     - False
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
     - -

Docstring
---------

.. code-block:: text

   Apply cuts on triggers in self.filteredclist. Call with filters=None,
   rejects=None, and reset_filters=True to clear filters
   
   PARAMETERS:
       filters: Dictionary with cuts in the form
                {parameter name: (vmin1, vmax1)}. Make vmin1 or vmax1
                -/+np.inf if you don't want to apply that
                List of allowed parameters:
                time: Offset of shift corrected trigger times from fiducial
                      start of central file
                snr: Signal to noise (properly corrected by PSD drift etc)
                cossnr: Cosine component of SNR
                sinsnr: Sine component of SNR
                calpha: Coefficients of basis functions in template bank,
                        Need to pass alpha
                alpha: Index into calpha (start with zero)
                If any of the above parameters is not set, the data is not
                filtered by that criterion
       rejects: Dictionary with rejection criteria in the form
                {parameter name: [(vmin1, vmax1),...]}, or
                {parameter name: (vmin1, vmax1)} if only one
       reset_filters: Boolean flag to indicate whether filters/rejects are
                      to be applied on top of existing ones
