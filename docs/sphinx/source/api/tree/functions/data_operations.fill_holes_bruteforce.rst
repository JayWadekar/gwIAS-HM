data_operations.fill_holes_bruteforce
=====================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def fill_holes_bruteforce(data, qmask, wt_filter_fd)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``data``
     - -
     - -
     - Array with strain data
   * - ``qmask``
     - -
     - -
     - Boolean mask with zeros at holes in unwhitened data
   * - ``wt_filter_fd``
     - -
     - -
     - Frequency domain whitening filter. Lives in space of rfft(len(data), dt)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Array of size len(data) with filled data

Docstring
---------

.. code-block:: text

   :param data: Array with strain data
   :param qmask: Boolean mask with zeros at holes in unwhitened data
   :param wt_filter_fd: Frequency domain whitening filter. Lives in space of
                        rfft(len(data), dt)
   :return: Array of size len(data) with filled data
