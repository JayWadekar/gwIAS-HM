template_bank_generator_HM.TemplateBank.split_whitened_wf_td
============================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Splits whitened time domain waveform into chunks with desired fractions of SNR^2, useful for vetoes

Signature
---------

.. code-block:: python

   def split_whitened_wf_td(whitened_wf_td, fractions = None, nchunk = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``whitened_wf_td``
     - -
     - -
     - Whitened time domain waveform
   * - ``fractions``
     - -
     - None
     - Array with fraction of SNR^2 in each chunk, we create len(array) + 1 chunks (last chunk makes it up to 1)
   * - ``nchunk``
     - -
     - 1
     - Number of equal SNR^2 chunks to split into, used only if fractions=None

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - nchunk x len(whitened_wf_td) array with time domain waveforms

Docstring
---------

.. code-block:: text

   Splits whitened time domain waveform into chunks with desired
   fractions of SNR^2, useful for vetoes
   :param whitened_wf_td: Whitened time domain waveform
   :param fractions: Array with fraction of SNR^2 in each chunk, we create
                     len(array) + 1 chunks (last chunk makes it up to 1)
   :param nchunk: Number of equal SNR^2 chunks to split into, used only
                  if fractions=None
   :return: nchunk x len(whitened_wf_td) array with time domain waveforms
