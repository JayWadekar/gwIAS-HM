utils.rm_suffix
===============

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Utility to change the extension of a path

Signature
---------

.. code-block:: python

   def rm_suffix(filepath, suffix = '.json', new_suffix = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``filepath``
     - -
     - -
     - String or an instance of pathlib.PosixPath
   * - ``suffix``
     - -
     - '.json'
     - Suffix to remove. Pass '.\\\*' to remove any existing extension
   * - ``new_suffix``
     - -
     - None
     - Suffix to add, if desired

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

   Utility to change the extension of a path
   :param filepath: String or an instance of pathlib.PosixPath
   :param suffix: Suffix to remove. Pass '.*' to remove any existing extension
   :param new_suffix: Suffix to add, if desired
   :return:
