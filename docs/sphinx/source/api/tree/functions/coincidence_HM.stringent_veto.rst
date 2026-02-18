coincidence_HM.stringent_veto
=============================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Apply stringent vetoes to optimized candidates for the edge case in which the best one in the bucket survives, but the coincident ones do not

Signature
---------

.. code-block:: python

   def stringent_veto(triggers, det_ind, trig_obj, min_veto_snr2 = None, group_duration = 0.1, origin = 0, n_cores = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``triggers``
     - -
     - -
     - n_triggers x 2 x row(processedclist) with coincident candidates
   * - ``det_ind``
     - -
     - -
     - Index of detector to veto (into 2nd dimension of trigger)
   * - ``trig_obj``
     - -
     - -
     - Instance of trig.TriggerList with processed data in the detector
   * - ``min_veto_snr2``
     - -
     - None
     - Veto triggers above this threshold
   * - ``group_duration``
     - -
     - 0.1
     - Group triggers every group_duration to save on setup FFTs
   * - ``origin``
     - -
     - 0
     - Origin to split the times relative to
   * - ``n_cores``
     - -
     - 1
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
     - 1. Boolean array marking whether the triggers pass stringent vetoes 2. Boolean array of shape n_triggers x (4 + len(split_chunks)) with zeros where glitch tests fired The indices correspond to 0: No chunks present 1: Overall chi-2 test 2: 2 + len(split_chunks): Split test 2 + len(split_chunks): No chunks with greater nchunk 3 + len(split_chunks): Chi^2 with greater nchunk

Docstring
---------

.. code-block:: text

   Apply stringent vetoes to optimized candidates for the edge case in which
   the best one in the bucket survives, but the coincident ones do not
   :param triggers:
       n_triggers x 2 x row(processedclist) with coincident candidates
   :param det_ind: Index of detector to veto (into 2nd dimension of trigger)
   :param trig_obj:
       Instance of trig.TriggerList with processed data in the detector
   :param min_veto_snr2: Veto triggers above this threshold
   :param group_duration:
       Group triggers every group_duration to save on setup FFTs
   :param origin: Origin to split the times relative to
   :param n_cores:
   :return:
       1. Boolean array marking whether the triggers pass stringent vetoes
       2. Boolean array of shape n_triggers x (4 + len(split_chunks))
           with zeros where glitch tests fired
           The indices correspond to
           0: No chunks present
           1: Overall chi-2 test
           2: 2 + len(split_chunks): Split test
           2 + len(split_chunks): No chunks with greater nchunk
           3 + len(split_chunks): Chi^2 with greater nchunk
