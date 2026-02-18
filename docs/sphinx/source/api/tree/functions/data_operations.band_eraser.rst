data_operations.band_eraser
===========================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Detects bandlimited excess power transients

Signature
---------

.. code-block:: python

   def band_eraser(strain_down, strain_wt_down, qmask_down, valid_mask, dt_down, notch_pars)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``strain_down``
     - -
     - -
     - Raw strain data, modifies in place (bands with high-power cells will be removed)
   * - ``strain_wt_down``
     - -
     - -
     - Modifies in place
   * - ``qmask_down``
     - -
     - -
     - Boolean mask with zeros at holes in unwhitened data
   * - ``valid_mask``
     - -
     - -
     - Boolean mask with zeros where we cannot trust whitened data
   * - ``dt_down``
     - -
     - -
     - Time interval between successive elements of data (s)
   * - ``notch_pars``
     - -
     - -
     - Parameters for notching loud lines from the strain

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - mask_stft: Mask with excess power bands removed

Docstring
---------

.. code-block:: text

   Detects bandlimited excess power transients
   :param strain_down: Raw strain data, modifies in place 
                       (bands with high-power cells will be removed)
   :param strain_wt_down: Modifies in place
   :param qmask_down: Boolean mask with zeros at holes in unwhitened data
   :param valid_mask:
       Boolean mask with zeros where we cannot trust whitened data
   :param dt_down: Time interval between successive elements of data (s)
   :param notch_pars: Parameters for notching loud lines from the strain
   :return: mask_stft: Mask with excess power bands removed
