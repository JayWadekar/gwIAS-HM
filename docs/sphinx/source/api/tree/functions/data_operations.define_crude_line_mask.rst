data_operations.define_crude_line_mask
======================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Marks lines in the PSD with zeros, errs on the side of marking lines

Signature
---------

.. code-block:: python

   def define_crude_line_mask(freqs, psds, nindpsd, sm_time = 1, fmin = params.FMIN_ANALYSIS)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``freqs``
     - -
     - -
     - -
   * - ``psds``
     - -
     - -
     - -
   * - ``nindpsd``
     - -
     - -
     - -
   * - ``sm_time``
     - -
     - 1
     - -
   * - ``fmin``
     - -
     - params.FMIN_ANALYSIS
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

   Marks lines in the PSD with zeros, errs on the side of marking lines
