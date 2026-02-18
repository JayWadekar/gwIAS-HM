data_operations.fill_holes_file
===============================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def fill_holes_file(data, qmask, qmask_prev, valid_mask, wt_filter_td, support_wt, erase_bands, band_erased)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``data``
     - -
     - -
     - Array of len(data) with strain data
   * - ``qmask``
     - -
     - -
     - Boolean mask of len(data) with zeros at holes in unwhitened data
   * - ``qmask_prev``
     - -
     - -
     - Boolean mask of len(data) with mask before previous instance of hole filling
   * - ``valid_mask``
     - -
     - -
     - Boolean mask with zeros marking where we cannot trust whitened data
   * - ``wt_filter_td``
     - -
     - -
     - Time domain whitening filter
   * - ``support_wt``
     - -
     - -
     - Time domain support of whitening filter (2 \\\* support_wt - 1 nonzero coeffs)
   * - ``erase_bands``
     - -
     - -
     - Boolean flag indicating whether we are erasing bands
   * - ``band_erased``
     - -
     - -
     - Boolean flag indicating whether we have erased bands

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Divides hole filling task into segments and passes to appropriate routines

Docstring
---------

.. code-block:: text

   :param data: Array of len(data) with strain data
   :param qmask:
       Boolean mask of len(data) with zeros at holes in unwhitened data
   :param qmask_prev:
       Boolean mask of len(data) with mask before previous instance of hole
       filling
   :param valid_mask:
       Boolean mask with zeros marking where we cannot trust whitened data
   :param wt_filter_td: Time domain whitening filter
   :param support_wt:
       Time domain support of whitening filter (2 * support_wt - 1
       nonzero coeffs)
   :param erase_bands: Boolean flag indicating whether we are erasing bands
   :param band_erased: Boolean flag indicating whether we have erased bands
   :return: Divides hole filling task into segments and passes to appropriate
            routines
