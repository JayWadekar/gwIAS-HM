template_bank_generator_HM.compute_overlap
==========================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Overlap between two waveforms

Signature
---------

.. code-block:: python

   def compute_overlap(wf1, wf2, fs, asds)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wf1``
     - -
     - -
     - ... x len(fs) array with frequency domain waveforms (units of 1/Hz), can be vector if n_wf = 1
   * - ``wf2``
     - -
     - -
     - ... x len(fs) array with frequency domain waveforms (units of 1/Hz), can be vector if n_wf = 1
   * - ``fs``
     - -
     - -
     - Array with regularly spaced frequencies (units of Hz)
   * - ``asds``
     - -
     - -
     - ASDs in units of 1/sqrt(Hz) at fs

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

   Overlap between two waveforms
   :param wf1: ... x len(fs) array with frequency domain waveforms
               (units of 1/Hz), can be vector if n_wf = 1
   :param wf2: ... x len(fs) array with frequency domain waveforms
               (units of 1/Hz), can be vector if n_wf = 1
   :param fs: Array with regularly spaced frequencies (units of Hz)
   :param asds: ASDs in units of 1/sqrt(Hz) at fs
   :return: Scalar or array of length n_wf with overlaps
