template_bank_generator_HM.summarize_prior_single_subbank
=========================================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

calpha_samples and calpha_test_set should have sample weights as their last column (the sample weights are corresponding to volume sensitivity in the case of the HM search)

Signature
---------

.. code-block:: python

   def summarize_prior_single_subbank(mb_key, subbank_ind, calpha_samples = None, calpha_test_set = None, calpha_samples_path = None, calpha_test_samples_path = None, force_smoothing_kernel = False, forced_sigma_noise = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``mb_key``
     - -
     - -
     - -
   * - ``subbank_ind``
     - -
     - -
     - -
   * - ``calpha_samples``
     - -
     - None
     - -
   * - ``calpha_test_set``
     - -
     - None
     - -
   * - ``calpha_samples_path``
     - -
     - None
     - -
   * - ``calpha_test_samples_path``
     - -
     - None
     - -
   * - ``force_smoothing_kernel``
     - -
     - False
     - -
   * - ``forced_sigma_noise``
     - -
     - None
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

   calpha_samples and calpha_test_set should have sample weights as their last column
   (the sample weights are corresponding to volume sensitivity in the case of the
    HM search)
