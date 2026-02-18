data_operations.fill_hole_consecutive
=====================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def fill_hole_consecutive(data, leftind, rightind, wt_filter_fd)

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
   * - ``leftind``
     - -
     - -
     - Left index of hole
   * - ``rightind``
     - -
     - -
     - Right index of hole
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
   :param leftind: Left index of hole
   :param rightind: Right index of hole
   :param wt_filter_fd: Frequency domain whitening filter. Lives in space of
                        rfft(len(data), dt)
   :return: Array of size len(data) with filled data
