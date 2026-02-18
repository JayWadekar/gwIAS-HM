template_bank_generator_HM.TemplateBank.gen_whitened_wfs_td
===========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Generates whitened and conditioned time-domain waveforms given basis coefficients, or input waveforms

Signature
---------

.. code-block:: python

   def gen_whitened_wfs_td(self, calpha = None, wfs_fd = None, orthogonalize = True, return_cov = False, **conditioning_kwargs)

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
   * - ``wfs_fd``
     - -
     - None
     - Frequency domain waveform(s) on self.fs_fft to whiten instead, using saved conditioning parameters (preferred over calpha) NOTE: wfs_fd should be linear free if you pass truncate=True
   * - ``orthogonalize``
     - -
     - True
     - Orthogonalizes the different mode wfs for calpha case
   * - ``return_cov``
     - -
     - False
     - Returns the covariance matrix between mode wfs
   * - ``\*\*conditioning_kwargs``
     - -
     - -
     - If known, pass conditioning parameters (useful when whitening different waveforms), defaults to those in the bank Can use multiple normfacs if needed

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_wf x n_modes x fftsize array of whitened TD waveforms (can be 2D for n_wf=1)

Docstring
---------

.. code-block:: text

   Generates whitened and conditioned time-domain waveforms given basis
   coefficients, or input waveforms
   :param calpha:
       n_wf x n_(basis elements needed) array with list of coefficients
       (can be vector for n_wf = 1). Defaults to the central waveform
   :param wfs_fd:
       Frequency domain waveform(s) on self.fs_fft to whiten instead,
       using saved conditioning parameters (preferred over calpha)
       NOTE: wfs_fd should be linear free if you pass truncate=True
   :param conditioning_kwargs:
       If known, pass conditioning parameters (useful when whitening
       different waveforms), defaults to those in the bank
       Can use multiple normfacs if needed
   :param orthogonalize: Orthogonalizes the different mode wfs for calpha case
   :param return_cov: Returns the covariance matrix between mode wfs
   :return: n_wf x n_modes x fftsize array of whitened TD waveforms
            (can be 2D for n_wf=1)
