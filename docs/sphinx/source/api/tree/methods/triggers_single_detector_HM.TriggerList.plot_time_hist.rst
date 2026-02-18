triggers_single_detector_HM.TriggerList.plot_time_hist
======================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Plots histogram of trigger time offsets from start of file (self.t0)

Signature
---------

.. code-block:: python

   def plot_time_hist(self, ax = None, bins = 1000, **hist_kwargs)

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
   * - ``bins``
     - -
     - 1000
     - Number of bins for histogram
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

   Plots histogram of trigger time offsets from start of file (self.t0)
   :param ax: Axis to plot into, if available
   :param bins: Number of bins for histogram
   :return: Axis with histogram, if none passed in
