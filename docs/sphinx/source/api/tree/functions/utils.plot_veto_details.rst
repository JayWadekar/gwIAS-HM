utils.plot_veto_details
=======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Make four-panel diagnostic plot for vetoes

Signature
---------

.. code-block:: python

   def plot_veto_details(tobj, candidate, signal_enhancement_factor = 1, dt_l = 10, dt_r = 6, nfft = 128, noverlap = None, use_HM = False, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``tobj``
     - -
     - -
     - Trigger object
   * - ``candidate``
     - -
     - -
     - Array with row of processed clist
   * - ``signal_enhancement_factor``
     - -
     - 1
     - -
   * - ``dt_l``
     - -
     - 10
     - -
   * - ``dt_r``
     - -
     - 6
     - -
   * - ``nfft``
     - -
     - 128
     - -
   * - ``noverlap``
     - -
     - None
     - -
   * - ``use_HM``
     - -
     - False
     - use triggers_single_detector_HM.py instead of triggers_single_detector.py
   * - ``\*\*kwargs``
     - -
     - -
     - Extra keyword arguments to pass to vetoes

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

   Make four-panel diagnostic plot for vetoes
   :param tobj: Trigger object
   :param candidate: Array with row of processed clist
   :param kwargs: Extra keyword arguments to pass to vetoes
   :param use_HM: use triggers_single_detector_HM.py instead of 
                   triggers_single_detector.py
