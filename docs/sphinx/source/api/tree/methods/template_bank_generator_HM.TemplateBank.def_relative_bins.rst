template_bank_generator_HM.TemplateBank.def_relative_bins
=========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Returns bin edges at which we evaluate relative waveforms

Signature
---------

.. code-block:: python

   def def_relative_bins(self, dcalphas, dt = params.DT_OPT, delta = 0.1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dcalphas``
     - -
     - -
     - Array with calpha spacing in the first few desired directions
   * - ``dt``
     - -
     - params.DT_OPT
     - Time shift to allow
   * - ``delta``
     - -
     - 0.1
     - Phase allowed to accumulate within each bin

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Array with bin edges within fs_basis

Docstring
---------

.. code-block:: text

   Returns bin edges at which we evaluate relative waveforms
   :param dcalphas:
       Array with calpha spacing in the first few desired directions
   :param dt: Time shift to allow
   :param delta: Phase allowed to accumulate within each bin
   :return: Array with bin edges within fs_basis
