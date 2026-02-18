triggers_single_detector_HM.TriggerList.get_bad_times
=====================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Finds bad times for the file, where it overproduces triggers due to widespread problems TODO: Fix this using preserve_max_snr?

Signature
---------

.. code-block:: python

   def get_bad_times(self, oversubscription = -1, maximum_interval = 0.1, rejection_interval = 25, snr2_thresh = 30, nperrun = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``oversubscription``
     - -
     - -1
     - Oversubscription factor / number of templates, pass -ve to skip
   * - ``maximum_interval``
     - -
     - 0.1
     - Interval (s) within which to count only the maximum SNR^2 trigger, needed to avoid penalizing events. Ensure that it is wide enough to catch all the friends
   * - ``rejection_interval``
     - -
     - 25
     - Interval (s) to reject based on total number of triggers within above SNR cut
   * - ``snr2_thresh``
     - -
     - 30
     - Threshold for maximum SNR2 to check overproduction above
   * - ``nperrun``
     - -
     - 1
     - Number of times this should fire per run

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Numpy array with list of times (s), invalid times are (t, t + rejection_interval) for t in result

Docstring
---------

.. code-block:: text

   Finds bad times for the file, where it overproduces triggers due to
   widespread problems
   TODO: Fix this using preserve_max_snr?
   :param oversubscription:
       Oversubscription factor / number of templates, pass -ve to skip
   :param maximum_interval:
       Interval (s) within which to count only the maximum SNR^2 trigger,
       needed to avoid penalizing events. Ensure that it is wide enough
       to catch all the friends
   :param rejection_interval:
       Interval (s) to reject based on total number of triggers within
       above SNR cut
   :param snr2_thresh:
       Threshold for maximum SNR2 to check overproduction above
   :param nperrun: Number of times this should fire per run
   :return:
       Numpy array with list of times (s), invalid times are
       (t, t + rejection_interval) for t in result
