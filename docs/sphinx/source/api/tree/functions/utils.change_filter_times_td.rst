utils.change_filter_times_td
============================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Converts a conditioned filter to a different time array by zero-padding in time-domain

Signature
---------

.. code-block:: python

   def change_filter_times_td(filter_td, orig_len, chop_len, pad_mode = 'center')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``filter_td``
     - -
     - -
     - n_filter x orig_len array with time-domain filters (can be vector for n_filter = 1)
   * - ``orig_len``
     - -
     - -
     - Length of original time series
   * - ``chop_len``
     - -
     - -
     - Length of different time series
   * - ``pad_mode``
     - -
     - 'center'
     - Where to pad, can be 'center', 'left' or 'right'

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

   Converts a conditioned filter to a different time array by
   zero-padding in time-domain
   :param filter_td:
       n_filter x orig_len array with time-domain filters
       (can be vector for n_filter = 1)
   :param orig_len: Length of original time series
   :param chop_len: Length of different time series
   :param pad_mode: Where to pad, can be 'center', 'left' or 'right'
   :return Time domain filter(s) living on chop_len
