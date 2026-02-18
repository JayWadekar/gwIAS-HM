template_bank_generator_HM.TemplateBank.gen_wfs_td_from_calpha
==============================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Generates time-domain waveforms given basis coefficients OK for FFT test but might not be for injection due to wraparound and truncation losses

Signature
---------

.. code-block:: python

   def gen_wfs_td_from_calpha(self, calpha = None, orthogonalize = True, return_cov = False, truncate = True, **td_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``calpha``
     - -
     - None
     - n_wf x n_(basis elements needed) array with list of coefficients (can be vector for n_wf = 1). Defaults to the central waveform
   * - ``orthogonalize``
     - -
     - True
     - Orthogonalizes the different mode wfs
   * - ``return_cov``
     - -
     - False
     - Returns the covariance matrix between mode wfs
   * - ``truncate``
     - -
     - True
     - -
   * - ``\*\*td_kwargs``
     - -
     - -
     - Extra arguments to gen_wfs_td_from_fd

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_wf x n_modes x fftsize array of TD waveforms (can be 2D for n_wf=1)

Docstring
---------

.. code-block:: text

   Generates time-domain waveforms given basis coefficients
   OK for FFT test but might not be for injection due to wraparound and
   truncation losses
   :param calpha:
       n_wf x n_(basis elements needed) array with list of coefficients
       (can be vector for n_wf = 1). Defaults to the central waveform
   :param td_kwargs: Extra arguments to gen_wfs_td_from_fd
   :param orthogonalize: Orthogonalizes the different mode wfs
   :param return_cov: Returns the covariance matrix between mode wfs
   :return:
       n_wf x n_modes x fftsize array of TD waveforms
       (can be 2D for n_wf=1)
