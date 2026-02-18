triggers_single_detector_HM.TriggerList.plot_bestfit_waveform
=============================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Plot waveforms in trigger(s) against the data

Signature
---------

.. code-block:: python

   def plot_bestfit_waveform(self, triggers, wfs_wt_cos_td = None, plot_injection = False, labels = None, ax = None, ref_time = None, individual_modes = False, physical_mode_ratio = False, data_plot_kwargs = {}, wf_plot_kwargs = {})

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``triggers``
     - -
     - -
     - Processedclists of triggers to plot (can be row if n_trigs = 1)
   * - ``wfs_wt_cos_td``
     - -
     - None
     - If known, array with cosine waveforms in rows (will infer from the bank otherwise)
   * - ``plot_injection``
     - -
     - False
     - Flag whether to plot the injected waveform
   * - ``labels``
     - -
     - None
     - If known, labels to add to the plot
   * - ``ax``
     - -
     - None
     - If known, axis to plot into
   * - ``ref_time``
     - -
     - None
     - -
   * - ``individual_modes``
     - -
     - False
     - -
   * - ``physical_mode_ratio``
     - -
     - False
     - -
   * - ``data_plot_kwargs``
     - -
     - {}
     - -
   * - ``wf_plot_kwargs``
     - -
     - {}
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
     - Axis that was plotted into, if none was passed in

Docstring
---------

.. code-block:: text

   Plot waveforms in trigger(s) against the data
   :param triggers:
       Processedclists of triggers to plot (can be row if n_trigs = 1)
   :param wfs_wt_cos_td:
       If known, array with cosine waveforms in rows
       (will infer from the bank otherwise)
   :param plot_injection: Flag whether to plot the injected waveform
   :param labels: If known, labels to add to the plot
   :param ax: If known, axis to plot into
   :return: Axis that was plotted into, if none was passed in
