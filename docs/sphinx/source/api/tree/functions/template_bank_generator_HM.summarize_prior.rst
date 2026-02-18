template_bank_generator_HM.summarize_prior
==========================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

This module creates the .pickle file for the template prior which is later used in ranking the candidates.

Signature
---------

.. code-block:: python

   def summarize_prior(folder, test_folder, name = None, force_smoothing_kernel = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``folder``
     - -
     - -
     - where to load calpha samples and physical parameters samples from.
   * - ``test_folder``
     - -
     - -
     - -
   * - ``name``
     - -
     - None
     - -
   * - ``force_smoothing_kernel``
     - -
     - False
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

   This module creates the .pickle file for the template prior which is later used in 
   ranking the candidates.
   :param folder: where to load calpha samples and physical parameters samples from.
