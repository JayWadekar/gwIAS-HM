data_operations.median_bias
===========================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Returns the bias of the median of a set of periodograms relative to the mean. See arXiv:gr-qc/0509116 Appendix B for details.

Signature
---------

.. code-block:: python

   def median_bias(n)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``n``
     - int
     - -
     - Numbers of periodograms being averaged.

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - float
     - Calculated bias.

Docstring
---------

.. code-block:: text

   Returns the bias of the median of a set of periodograms relative to
   the mean.
   See arXiv:gr-qc/0509116 Appendix B for details.
   Parameters
   ----------
   n : int
       Numbers of periodograms being averaged.
   Returns
   -------
   bias : float
       Calculated bias.
