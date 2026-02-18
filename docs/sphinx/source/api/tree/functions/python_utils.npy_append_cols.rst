python_utils.npy_append_cols
============================

Back to :doc:`Module page <../modules/python_utils>`

Summary
-------

append columns to array from .npy file at path infile if outfile_new given, write the result there; otherwise rewrite infile_npy with appended columns

Signature
---------

.. code-block:: python

   def npy_append_cols(infile_npy, add_cols, outfile_npy = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``infile_npy``
     - -
     - -
     - -
   * - ``add_cols``
     - -
     - -
     - -
   * - ``outfile_npy``
     - -
     - None
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

   append columns to array from .npy file at path infile
     if outfile_new given, write the result there;
     otherwise rewrite infile_npy with appended columns
