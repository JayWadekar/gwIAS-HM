utils.is_close_to
=================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def is_close_to(t_1, t_2, t_1_start = None, eps = 4, return_close = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``t_1``
     - -
     - -
     - List of reference times (ends of waveforms)
   * - ``t_2``
     - -
     - -
     - Time, or list of times that we are checking against the reference times
   * - ``t_1_start``
     - -
     - None
     - if available, list of start times of waveforms
   * - ``eps``
     - -
     - 4
     - Tolerance to mark associated triggers
   * - ``return_close``
     - -
     - False
     - Flag to return len(t_2) x len(t_1) boolean matrix indicating closeness

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Boolean flag / array of flags listing whether time / list of times in t_2 is close to any of the t_1 if return_close, returns adjacency matrix too

Docstring
---------

.. code-block:: text

   :param t_1: List of reference times (ends of waveforms)
   :param t_2:
       Time, or list of times that we are checking against the reference times
   :param t_1_start: if available, list of start times of waveforms
   :param eps: Tolerance to mark associated triggers
   :param return_close:
       Flag to return len(t_2) x len(t_1) boolean matrix indicating closeness
   :return:
       Boolean flag / array of flags listing whether time / list of times in
       t_2 is close to any of the t_1
       if return_close, returns adjacency matrix too
