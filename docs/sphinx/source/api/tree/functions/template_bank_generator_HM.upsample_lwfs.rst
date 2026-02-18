template_bank_generator_HM.upsample_lwfs
========================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Upsample given lwfs to an output frequency grid Note: Amplitudes are linearly interpolated, with zeros outside the range Pad phases with edge values to avoid discontinuities at the edges

Signature
---------

.. code-block:: python

   def upsample_lwfs(lwfs, fs_in, fs_out, phase_only = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``lwfs``
     - -
     - -
     - ... x len(fs_in) array with log of waveforms in frequency domain (make sure to pass unwrapped phase)
   * - ``fs_in``
     - -
     - -
     - Array of size(fs_in) with frequencies
   * - ``fs_out``
     - -
     - -
     - Array of size(fs_out) with frequencies
   * - ``phase_only``
     - -
     - False
     - Flag indicating whether to compute and return only the phases

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - ... x len(fs_out) array with log of upsampled waveforms (phases if phase_only is True)

Docstring
---------

.. code-block:: text

   Upsample given lwfs to an output frequency grid
   Note: Amplitudes are linearly interpolated, with zeros outside the range
         Pad phases with edge values to avoid discontinuities at the edges
   :param lwfs:
       ... x len(fs_in) array with log of waveforms in frequency domain
       (make sure to pass unwrapped phase)
   :param fs_in: Array of size(fs_in) with frequencies
   :param fs_out: Array of size(fs_out) with frequencies
   :param phase_only:
       Flag indicating whether to compute and return only the phases
   :return:
       ... x len(fs_out) array with log of upsampled waveforms
       (phases if phase_only is True)
