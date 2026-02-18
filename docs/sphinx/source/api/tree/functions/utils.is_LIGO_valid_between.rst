utils.is_LIGO_valid_between
===========================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Take two gps times t1 and t2 and return: 0 if the hole chunk from t1 to t2 is invalid 1 if it is partially valid 2 if it is fully valid where valid means that both H and L have all the ['DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3'] flags True. (e.g. for assessing whether a waveform between t1 and t2 had the right to be found or not)

Signature
---------

.. code-block:: python

   def is_LIGO_valid_between(t1, t2)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``t1``
     - -
     - -
     - -
   * - ``t2``
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
     - -

Docstring
---------

.. code-block:: text

   Take two gps times t1 and t2 and return:
       0 if the hole chunk from t1 to t2 is invalid
       1 if it is partially valid
       2 if it is fully valid
   where valid means that both H and L have all the
   ['DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3']
   flags True.
   (e.g. for assessing whether a waveform between t1 and t2
   had the right to be found or not)
