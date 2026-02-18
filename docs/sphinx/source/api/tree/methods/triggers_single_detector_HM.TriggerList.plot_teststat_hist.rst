triggers_single_detector_HM.TriggerList.plot_teststat_hist
==========================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Plots normalized overall histogram of test statistic

Signature
---------

.. code-block:: python

   def plot_teststat_hist(self, ax = None, undo_psd_drift = False, plot_chi2 = True, **hist_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``ax``
     - -
     - None
     - Axis to plot into, if available
   * - ``undo_psd_drift``
     - -
     - False
     - Flag indicating whether to undo psd drift
   * - ``plot_chi2``
     - -
     - True
     - Flag to plot chi2 line
   * - ``\*\*hist_kwargs``
     - -
     - -
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
     - Axis with histogram, if none passed in

Docstring
---------

.. code-block:: text

   Plots normalized overall histogram of test statistic
   :param ax: Axis to plot into, if available
   :param undo_psd_drift: Flag indicating whether to undo psd drift
   :param plot_chi2: Flag to plot chi2 line
   :return: Axis with histogram, if none passed in
