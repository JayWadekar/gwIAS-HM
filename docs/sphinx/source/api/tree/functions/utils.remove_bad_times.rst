utils.remove_bad_times
======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Removes elements of dat that are in the same bucket as in bad_time_list

Signature
---------

.. code-block:: python

   def remove_bad_times(bad_time_list, time_list, time_shift_tol, *dat)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``bad_time_list``
     - -
     - -
     - List of bad times (s)
   * - ``time_list``
     - -
     - -
     - List of times that elements in dat correspond to (s)
   * - ``time_shift_tol``
     - -
     - -
     - Tolerance for buckets (s)
   * - ``\*dat``
     - -
     - -
     - Any number of lists that need to be cleaned of bad times

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - dat, with bad times cleaned. If dat is a single list, returns list instead of list of lists

Docstring
---------

.. code-block:: text

   Removes elements of dat that are in the same bucket as in bad_time_list
   :param bad_time_list: List of bad times (s)
   :param time_list: List of times that elements in dat correspond to (s)
   :param time_shift_tol: Tolerance for buckets (s)
   :param dat: Any number of lists that need to be cleaned of bad times
   :return:
       dat, with bad times cleaned. If dat is a single list, returns list
       instead of list of lists
