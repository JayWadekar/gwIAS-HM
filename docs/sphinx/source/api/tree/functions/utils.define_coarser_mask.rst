utils.define_coarser_mask
=========================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def define_coarser_mask(freqs_in, mask_freqs_in, freqs_out)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``freqs_in``
     - -
     - -
     - Array with fine frequency grid
   * - ``mask_freqs_in``
     - -
     - -
     - Boolean mask on fine frequency grid
   * - ``freqs_out``
     - -
     - -
     - Array with coarse frequency grid

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Mask on freqs_out with product of entries of mask_freqs_in in the relevant range

Docstring
---------

.. code-block:: text

   :param freqs_in: Array with fine frequency grid
   :param mask_freqs_in: Boolean mask on fine frequency grid
   :param freqs_out: Array with coarse frequency grid
   :return:
       Mask on freqs_out with product of entries of mask_freqs_in in the
       relevant range
