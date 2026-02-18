utils.splitarray
================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Splits a single or multi-dimensional array into sub-arrays based on a coordinate

Signature
---------

.. code-block:: python

   def splitarray(dat, splitvec, interval, axis = 0, return_split_keys = False, origin = 0)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dat``
     - -
     - -
     - List/array to split (can be multidimensional, in which case it has to be a numpy array)
   * - ``splitvec``
     - -
     - -
     - Numpy vector to split array dimension by
   * - ``interval``
     - -
     - -
     - Length of buckets for values in splitvec
   * - ``axis``
     - -
     - 0
     - (>0) Axis to split dat along, dimension should match len(splitvec)
   * - ``return_split_keys``
     - -
     - False
     - Flag indicating whether to return keys that we split according to
   * - ``origin``
     - -
     - 0
     - Split according to offsets from this origin

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - If return_split_keys is True, 1. Keys split according to 2. List of subarrays of dat split along axis according to values in splitvec every interval else, only the second one

Docstring
---------

.. code-block:: text

   Splits a single or multi-dimensional array into sub-arrays based on a
   coordinate
   :param dat:
       List/array to split (can be multidimensional, in which case it has to
       be a numpy array)
   :param splitvec: Numpy vector to split array dimension by
   :param interval: Length of buckets for values in splitvec
   :param axis:
       (>0) Axis to split dat along, dimension should match len(splitvec)
   :param return_split_keys:
       Flag indicating whether to return keys that we split according to
   :param origin: Split according to offsets from this origin
   :return:
       If return_split_keys is True,
       1. Keys split according to
       2. List of subarrays of dat split along axis according to values in
          splitvec every interval
       else, only the second one
