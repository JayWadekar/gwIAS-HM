template_bank_generator_HM.TemplateBank.orthogonalize_wfs
=========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Orthogonalizes the different higher modes in the wf (useful for later appropriately calculating inner product of data with diff modes)

Signature
---------

.. code-block:: python

   def orthogonalize_wfs(self, wfs, weights)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wfs``
     - -
     - -
     - n_modes (=3) x n_freq_basis wf in the Fourier domain
   * - ``weights``
     - -
     - -
     - n_freq_basis for weights for different freq bins

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_modes x len(fs_out) wf and upper triangular elements of the covariance matrix

Docstring
---------

.. code-block:: text

   Orthogonalizes the different higher modes in the wf
   (useful for later appropriately calculating inner product
    of data with diff modes)
   :param wfs: n_modes (=3) x n_freq_basis wf in the Fourier domain
   :param weights: n_freq_basis for weights for different freq bins
   :return:
       n_modes x len(fs_out) wf and upper triangular elements of 
       the covariance matrix
