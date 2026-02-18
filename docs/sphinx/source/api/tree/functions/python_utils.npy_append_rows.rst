python_utils.npy_append_rows
============================

Back to :doc:`Module page <../modules/python_utils>`

Summary
-------

append rows to array from .npy file at path infile if outfile_new given, write the result there; otherwise rewrite infile_npy with appended rows

Signature
---------

.. code-block:: python

   def npy_append_rows(infile_npy, add_rows, outfile_npy = None)

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
   * - ``add_rows``
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

   append rows to array from .npy file at path infile
     if outfile_new given, write the result there;
     otherwise rewrite infile_npy with appended rows
