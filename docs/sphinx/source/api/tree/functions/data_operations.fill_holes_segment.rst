data_operations.fill_holes_segment
==================================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Fill holes in a segment of data

Signature
---------

.. code-block:: python

   def fill_holes_segment(data_seg, qmask_seg, valid_mask_seg, wt_filter_td, support_wt, erase_bands, band_erased, qmask_prev_seg)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``data_seg``
     - -
     - -
     - Segment of whitened strained data
   * - ``qmask_seg``
     - -
     - -
     - Boolean mask with zeros at holes in unwhitened data
   * - ``valid_mask_seg``
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
   * - ``qmask_prev_seg``
     - -
     - -
     - Boolean mask with zeros at holes in unwhitened data with the previous Boolean mask

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Inpaints holes in data in place, expands holes and quality mask if needed, and marks valid_mask if we failed

Docstring
---------

.. code-block:: text

   Fill holes in a segment of data
   :param data_seg: Segment of whitened strained data
   :param qmask_seg: Boolean mask with zeros at holes in unwhitened data
   :param valid_mask_seg: Boolean mask with zeros marking where we cannot
                          trust whitened data
   :param wt_filter_td: Time domain whitening filter
   :param support_wt: Time domain support of whitening filter
                      (2 * support_wt - 1 nonzero coeffs)
   :param erase_bands: Boolean flag indicating whether we are erasing bands
   :param band_erased: Boolean flag indicating whether we have erased bands
   :param qmask_prev_seg: Boolean mask with zeros at holes in unwhitened data
                           with the previous Boolean mask
   :return: Inpaints holes in data in place, expands holes and quality mask
            if needed, and marks valid_mask if we failed
