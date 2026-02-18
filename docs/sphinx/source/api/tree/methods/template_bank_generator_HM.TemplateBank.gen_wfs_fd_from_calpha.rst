template_bank_generator_HM.TemplateBank.gen_wfs_fd_from_calpha
==============================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Generates waveforms from given coefficients Note: At fs_out <= min(fs_basis) or >= max(fs_basis), the output amplitudes are zero

Signature
---------

.. code-block:: python

   def gen_wfs_fd_from_calpha(self, calpha = None, fs_out = None, orthogonalize = False, return_cov = False, log = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``calpha``
     - -
     - None
     - n_wf x n_(basis elements needed) array with list of coefficients (can be a vector for n_wf = 1). Defaults to the central waveform
   * - ``fs_out``
     - -
     - None
     - Array of output frequencies. None indicates fs_out = fs_basis
   * - ``orthogonalize``
     - -
     - False
     - Only works when fs_out is self.fs_fft Note: better to orthogonalize wfs after whitening, rather than at this stage
   * - ``return_cov``
     - -
     - False
     - -
   * - ``log``
     - -
     - False
     - FLag to return the log of the waveform instead

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_wf x n_modes x len(fs_out) complex array with waveforms at fs_out (can be 2D for n_wf=1)

Docstring
---------

.. code-block:: text

   Generates waveforms from given coefficients
   Note: At fs_out <= min(fs_basis) or >= max(fs_basis), the output
   amplitudes are zero
   :param calpha:
       n_wf x n_(basis elements needed) array with list of coefficients
       (can be a vector for n_wf = 1). Defaults to the central waveform
   :param fs_out:
       Array of output frequencies. None indicates fs_out = fs_basis
   :param orthogonalize: Only works when fs_out is self.fs_fft
       Note: better to orthogonalize wfs after whitening, rather than at
       this stage
   :param log: FLag to return the log of the waveform instead
   :return:
       n_wf x n_modes x len(fs_out) complex array with waveforms at fs_out
       (can be 2D for n_wf=1)
