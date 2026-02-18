data_operations.update_masks
============================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Convenience function to update a mask in a range, record where we nulled, and avoid some regions if needed

Signature
---------

.. code-block:: python

   def update_masks(ileft, iright, qmask = None, outlier_mask = None, mask_save = None, mask_temp = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``ileft``
     - -
     - -
     - Index of left edge of region to null
   * - ``iright``
     - -
     - -
     - Index of right edge of region to null (not inclusive)
   * - ``qmask``
     - -
     - None
     - Global boolean mask to update by nulling, if needed
   * - ``outlier_mask``
     - -
     - None
     - Boolean array to null to record where we ended up nulling this round, if needed
   * - ``mask_save``
     - -
     - None
     - Boolean array with zeros at indices that we want to exempt from nulling
   * - ``mask_temp``
     - -
     - None
     - Boolean array for working memory

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Modifies input masks in place, returns number of indices updated

Docstring
---------

.. code-block:: text

   Convenience function to update a mask in a range, record where we nulled,
   and avoid some regions if needed
   :param ileft: Index of left edge of region to null
   :param iright: Index of right edge of region to null (not inclusive)
   :param qmask: Global boolean mask to update by nulling, if needed
   :param outlier_mask:
       Boolean array to null to record where we ended up nulling this round,
       if needed
   :param mask_save:
       Boolean array with zeros at indices that we want to exempt from nulling
   :param mask_temp: Boolean array for working memory
   :return: Modifies input masks in place, returns number of indices updated
