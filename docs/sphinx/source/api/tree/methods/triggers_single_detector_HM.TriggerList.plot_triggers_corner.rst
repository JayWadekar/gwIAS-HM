triggers_single_detector_HM.TriggerList.plot_triggers_corner
============================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Make corner plot of parameters of triggers

Signature
---------

.. code-block:: python

   def plot_triggers_corner(trig = None, filteredclist = None, indlist = (0, 1, 7, 8), t0 = None, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trig``
     - -
     - None
     - Trigger object whose filteredclist to use
   * - ``filteredclist``
     - -
     - None
     - filteredclist to use
   * - ``indlist``
     - -
     - (0, 1, 7, 8)
     - Tuple of indices into self.filteredclist to plot
   * - ``t0``
     - -
     - None
     - Origin for time, if known (defaults to t0 of file)
   * - ``\*\*kwargs``
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
     - Figure with plot

Docstring
---------

.. code-block:: text

   Make corner plot of parameters of triggers
   :param trig: Trigger object whose filteredclist to use
   :param filteredclist: filteredclist to use
   :param indlist: Tuple of indices into self.filteredclist to plot
   :param t0: Origin for time, if known (defaults to t0 of file)
   :return: Figure with plot
