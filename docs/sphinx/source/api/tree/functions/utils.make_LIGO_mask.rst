utils.make_LIGO_mask
====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Run this only once to make the file /data/bzackay/GW/LIGO_holes.npy

Signature
---------

.. code-block:: python

   def make_LIGO_mask(mask_keys = ('DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3'), tmin = TMIN_O1, tmax = TMAX_O2, save = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``mask_keys``
     - -
     - ('DATA', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3')
     - -
   * - ``tmin``
     - -
     - TMIN_O1
     - -
   * - ``tmax``
     - -
     - TMAX_O2
     - -
   * - ``save``
     - -
     - False
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

   Run this only once to make the file /data/bzackay/GW/LIGO_holes.npy
