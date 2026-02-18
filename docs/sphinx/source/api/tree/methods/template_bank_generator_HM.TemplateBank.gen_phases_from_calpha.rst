template_bank_generator_HM.TemplateBank.gen_phases_from_calpha
==============================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Generates phases of all modes from given coefficients

Signature
---------

.. code-block:: python

   def gen_phases_from_calpha(self, calpha = None, fs_out = None)

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
     - Array of output frequencies. None indicates fs_out = fs_basis Can also be iterable of arrays for different modes

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - if fs_out is same for all modes: n_wf x 3 x len(fs_out) real array with phases at fs_out else: n_wf x [len(f) in fs_out]

Docstring
---------

.. code-block:: text

   Generates phases of all modes from given coefficients
   :param calpha:
       n_wf x n_(basis elements needed) array with list of coefficients
       (can be vector for n_wf = 1). Defaults to the central waveform
   :param fs_out:
       Array of output frequencies. None indicates fs_out = fs_basis
       Can also be iterable of arrays for different modes
   :return:
       if fs_out is same for all modes:
           n_wf x 3 x len(fs_out) real array with phases at fs_out
       else:
           n_wf x [len(f) in fs_out]
