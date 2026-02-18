template_bank_generator_HM.TemplateBank.gen_phase_mismatch
==========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Has not been updated by Jay. Saves SNR degradation as a function of frequency for waveforms in bank. Note: Requires that bank was loaded with load_lwfs=True

Signature
---------

.. code-block:: python

   def gen_phase_mismatch(self, nwf = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``nwf``
     - -
     - None
     - Number of waveforms to get degradation for

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_wf x len(fs_basis) array with weight \\\* (angle - angle_calpha)

Docstring
---------

.. code-block:: text

   Has not been updated by Jay.
   Saves SNR degradation as a function of frequency for waveforms in
   bank. Note: Requires that bank was loaded with load_lwfs=True
   :param nwf: Number of waveforms to get degradation for
   :return: n_wf x len(fs_basis) array with weight * (angle - angle_calpha)
