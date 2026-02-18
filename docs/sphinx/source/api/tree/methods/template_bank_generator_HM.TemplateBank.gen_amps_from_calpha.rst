template_bank_generator_HM.TemplateBank.gen_amps_from_calpha
============================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Generates amplitudes of all modes from given calphas

Signature
---------

.. code-block:: python

   def gen_amps_from_calpha(self, calpha = None, fs_out = None)

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
   * - ``fs_out``
     - -
     - None
     - Array of output frequencies. None indicates fs_out = fs_basis

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_wf x 3 x len(fs_out) real array with amplitudes at fs_out

Docstring
---------

.. code-block:: text

   Generates amplitudes of all modes from given calphas
   :param calpha:
       n_wf x n_(basis elements needed) array with list of coefficients
       (can be vector for n_wf = 1). Defaults to the central waveform
   :param fs_out:
       Array of output frequencies. None indicates fs_out = fs_basis
   :return: n_wf x 3 x len(fs_out) real array with amplitudes at fs_out
