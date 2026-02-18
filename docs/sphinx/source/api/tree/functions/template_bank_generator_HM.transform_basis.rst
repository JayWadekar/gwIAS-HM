template_bank_generator_HM.transform_basis
==========================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Returns components of waveforms in the given basis

Signature
---------

.. code-block:: python

   def transform_basis(phases, avg_phase_evolution, basis, wts, fs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``phases``
     - -
     - -
     - n_wf x len(fs) complex array with unwrapped phases of frequency domain waveforms (note it needs to be a 2d array)
   * - ``avg_phase_evolution``
     - -
     - -
     - len(fs) array with average phase evolution \\\* weights
   * - ``basis``
     - -
     - -
     - n_basis x len(fs) array with orthonormal basis shapes
   * - ``wts``
     - -
     - -
     - Array of size len(fs) with inner product weights to convert wfs into basis
   * - ``fs``
     - -
     - -
     - Array with frequencies (Hz)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_wf x n_basis array with components of each waveform along each basis element

Docstring
---------

.. code-block:: text

   Returns components of waveforms in the given basis
   :param phases:
       n_wf x len(fs) complex array with unwrapped phases of
       frequency domain waveforms (note it needs to be a 2d array)
   :param avg_phase_evolution:
       len(fs) array with average phase evolution * weights
   :param basis: n_basis x len(fs) array with orthonormal basis shapes
   :param wts:
       Array of size len(fs) with inner product weights to convert
       wfs into basis
   :param fs: Array with frequencies (Hz)
   :return:
       n_wf x n_basis array with components of each waveform along each basis
       element
