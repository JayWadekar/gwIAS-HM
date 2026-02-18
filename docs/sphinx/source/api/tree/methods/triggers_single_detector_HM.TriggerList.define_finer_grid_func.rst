triggers_single_detector_HM.TriggerList.define_finer_grid_func
==============================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def define_finer_grid_func(self, dcalpha_coarse = None, trim_dims = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dcalpha_coarse``
     - -
     - None
     - calpha_spacing of coarse grid
   * - ``trim_dims``
     - -
     - True
     - Flag indicating whether to cut dimensions at self.dcalphas

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Function that returns calphas in a finer grid around a given calpha 2. Array with spacings of finer grid in all dimensions, used for safeties

Docstring
---------

.. code-block:: text

   :param dcalpha_coarse: calpha_spacing of coarse grid
   :param trim_dims:
       Flag indicating whether to cut dimensions at self.dcalphas
   :return:
       1. Function that returns calphas in a finer grid around a given
          calpha
       2. Array with spacings of finer grid in all dimensions, used for
          safeties
