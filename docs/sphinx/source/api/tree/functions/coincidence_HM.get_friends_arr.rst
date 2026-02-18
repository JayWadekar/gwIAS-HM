coincidence_HM.get_friends_arr
==============================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Groups triggers in clist into buckets every time_tol, and retains only those high enough relative to the maximum in each bucket

Signature
---------

.. code-block:: python

   def get_friends_arr(clist, time_tol, score_reduction_max, origin = 0)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``clist``
     - -
     - -
     - Processedclist
   * - ``time_tol``
     - -
     - -
     - Width of each bucket in time (s)
   * - ``score_reduction_max``
     - -
     - -
     - Allow a reduction in SNR^2 of max \\\* params.MAX_FRIEND_DEGRADE_SNR2 + score_reduction_max when keeping friends
   * - ``origin``
     - -
     - 0
     - Origin for splitting the times relative to

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - List of processedclists of friends, one for each bucket

Docstring
---------

.. code-block:: text

   Groups triggers in clist into buckets every time_tol, and retains only
   those high enough relative to the maximum in each bucket
   :param clist: Processedclist
   :param time_tol: Width of each bucket in time (s)
   :param score_reduction_max:
       Allow a reduction in SNR^2 of max * params.MAX_FRIEND_DEGRADE_SNR2 +
       score_reduction_max when keeping friends
   :param origin: Origin for splitting the times relative to
   :return: List of processedclists of friends, one for each bucket
