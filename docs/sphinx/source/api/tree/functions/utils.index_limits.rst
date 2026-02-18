utils.index_limits
==================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Function that decides which indices to include in the average such that we always average window_size indices to avoid \\\`regression-to-mean' artifacts due to fewer samples near holes

Signature
---------

.. code-block:: python

   def index_limits(window_index, extra_req, jump, window_size, valid_mask)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``window_index``
     - -
     - -
     - Index of window to decide limits for
   * - ``extra_req``
     - -
     - -
     - Number of extra indices to pull in
   * - ``jump``
     - -
     - -
     - Jump in indices between window starts
   * - ``window_size``
     - -
     - -
     - Size of window desired, after respecting the valid mask
   * - ``valid_mask``
     - -
     - -
     - Valid mask deciding which entries are to kept

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - left and right limits of window with required number of entries

Docstring
---------

.. code-block:: text

   Function that decides which indices to include in the average
   such that we always average window_size indices to avoid
   `regression-to-mean' artifacts due to fewer samples near holes
   :param window_index: Index of window to decide limits for
   :param extra_req: Number of extra indices to pull in
   :param jump: Jump in indices between window starts
   :param window_size: Size of window desired, after respecting the valid mask
   :param valid_mask: Valid mask deciding which entries are to kept
   :return: left and right limits of window with required number of entries
