utils.extract_filename
======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Extracts the filename from a string representation of a buffer object

Signature
---------

.. code-block:: python

   def extract_filename(fname)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fname``
     - -
     - -
     - String

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Filename if found, or the original string

Docstring
---------

.. code-block:: text

   Extracts the filename from a string representation of a buffer object
   :param fname: String
   :return: Filename if found, or the original string
