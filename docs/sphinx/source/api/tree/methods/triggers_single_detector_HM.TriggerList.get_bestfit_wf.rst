triggers_single_detector_HM.TriggerList.get_bestfit_wf
======================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def get_bestfit_wf(self, trigger, wf_wt_cos_td = None, individual_modes = False, physical_mode_ratio = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - -
     - Row of processedclist
   * - ``wf_wt_cos_td``
     - -
     - None
     - If known, the cosine whitened waveform (will infer from the template bank otherwise)
   * - ``individual_modes``
     - -
     - False
     - Return [22,33,44] separately instead of their sum
   * - ``physical_mode_ratio``
     - -
     - True
     - require A_33/A_22 and A_44/A_22 to be in the physically allowed region

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Array of length support_whitened_wf with bestfit waveform

Docstring
---------

.. code-block:: text

   :param trigger: Row of processedclist
   :param wf_wt_cos_td:
       If known, the cosine whitened waveform
       (will infer from the template bank otherwise)
   :param individual_modes: Return [22,33,44] separately instead of 
       their sum
   :param physical_mode_ratio: require A_33/A_22 and A_44/A_22 to be in
                               the physically allowed region
   :return: Array of length support_whitened_wf with bestfit waveform
