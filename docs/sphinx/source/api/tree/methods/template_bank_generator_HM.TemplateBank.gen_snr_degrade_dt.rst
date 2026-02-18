template_bank_generator_HM.TemplateBank.gen_snr_degrade_dt
==========================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Has not been updated by Jay. Computes SNR degradation for a number of random waveforms shifted by a single TD index, used in deciding the time resolution to use for the matched filtering

Signature
---------

.. code-block:: python

   def gen_snr_degrade_dt(self, n_wf = 100)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``n_wf``
     - -
     - 100
     - Number of random waveforms

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Degradation factor for n_wf random waveforms

Docstring
---------

.. code-block:: text

   Has not been updated by Jay.
   Computes SNR degradation for a number of random waveforms shifted by a
   single TD index, used in deciding the time resolution to use for the
   matched filtering
   :param n_wf: Number of random waveforms
   :return: Degradation factor for n_wf random waveforms
