triggers_single_detector_HM.TriggerList.prepare_subset_for_triggers
===================================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Prepare to compute everything on a reduced set of data to save time for BH-like waveforms. Defines subset of data, performs relevant FFTs, and stores in class variables Note that if relevant_index + right_inds_scores goes past the edge of the data, then we will have zeros in all the elements for which we don't have data

Signature
---------

.. code-block:: python

   def prepare_subset_for_triggers(self, relevant_index, left_inds_scores, right_inds_scores, zero_pad = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``relevant_index``
     - -
     - -
     - Index into self.time for trigger score (right edge of waveform) If 0, ensure that len(self.time) is passed instead
   * - ``left_inds_scores``
     - -
     - -
     - Left indices in scores to guarantee w.r.t relevant index, excluding relevant index
   * - ``right_inds_scores``
     - -
     - -
     - Right indices in scores to guarantee w.r.t relevant index, excluding relevant index
   * - ``zero_pad``
     - -
     - True
     - Flag indicating whether to zero pad, or pad with existing data

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

   Prepare to compute everything on a reduced set of data to save time
   for BH-like waveforms. Defines subset of data, performs relevant FFTs,
   and stores in class variables
   Note that if relevant_index + right_inds_scores goes past the edge of
   the data, then we will have zeros in all the elements for which we
   don't have data
   :param relevant_index:
       Index into self.time for trigger score (right edge of waveform)
       If 0, ensure that len(self.time) is passed instead
   :param left_inds_scores:
       Left indices in scores to guarantee w.r.t relevant index,
       excluding relevant index
   :param right_inds_scores:
       Right indices in scores to guarantee w.r.t relevant index,
       excluding relevant index
   :param zero_pad:
       Flag indicating whether to zero pad, or pad with existing data
