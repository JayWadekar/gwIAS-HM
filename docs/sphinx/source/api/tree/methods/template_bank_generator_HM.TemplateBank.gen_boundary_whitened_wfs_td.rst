template_bank_generator_HM.TemplateBank.gen_boundary_whitened_wfs_td
====================================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Generates boundary whitened and conditioned time-domain waveforms

Signature
---------

.. code-block:: python

   def gen_boundary_whitened_wfs_td(self, ncalpha = 3)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``ncalpha``
     - -
     - 3
     - Number of calphas for which to explore bounds

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 3\\\*\\\*n_calpha \\\* fftsize array containing boundary whitened TD waveforms

Docstring
---------

.. code-block:: text

   Generates boundary whitened and conditioned time-domain waveforms
   :param ncalpha: Number of calphas for which to explore bounds
   :return: 3**n_calpha * fftsize array containing boundary whitened TD
            waveforms
