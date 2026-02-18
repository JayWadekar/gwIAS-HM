template_bank_generator_HM.compute_snr_efficiency
=================================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Computes factor by which overlap degrades = SNR efficiency

Signature
---------

.. code-block:: python

   def compute_snr_efficiency(wf_true, wf_temp, fs, asds)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wf_true``
     - -
     - -
     - n_wf x len(fs) array w/ true frequency domain waveforms (units of 1/Hz), can be vector if n_wf = 1
   * - ``wf_temp``
     - -
     - -
     - n_wf x len(fs) array w/ template frequency domain waveforms (units of 1/Hz), can be vector if n_wf = 1
   * - ``fs``
     - -
     - -
     - Array with regularly spaced frequencies (units of Hz)
   * - ``asds``
     - -
     - -
     - ASDs in units of 1/sqrt(Hz) at fs. Take care that there are no zero or very small values

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Scalar or array of length n_wf with overlaps

Docstring
---------

.. code-block:: text

   Computes factor by which overlap degrades = SNR efficiency
   :param wf_true: n_wf x len(fs) array w/ true frequency domain waveforms
                   (units of 1/Hz), can be vector if n_wf = 1
   :param wf_temp: n_wf x len(fs) array w/ template frequency domain waveforms
                   (units of 1/Hz), can be vector if n_wf = 1
   :param fs: Array with regularly spaced frequencies (units of Hz)
   :param asds: ASDs in units of 1/sqrt(Hz) at fs. Take care that there are no
                zero or very small values
   :return: Scalar or array of length n_wf with overlaps
