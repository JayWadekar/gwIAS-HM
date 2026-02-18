utils.delete_hdf5_datasets
==========================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Convenience function to prune some leaves from a hdf5 file

Signature
---------

.. code-block:: python

   def delete_hdf5_datasets(source, h5path, leaves)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``source``
     - -
     - -
     - Path to the hdf5 file, or the file object itself
   * - ``h5path``
     - -
     - -
     - String, or an iterable of strings to reach the parent of the leaves
   * - ``leaves``
     - -
     - -
     - String, or iterable of strings with names of leaves to delete

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

   Convenience function to prune some leaves from a hdf5 file
   :param source: Path to the hdf5 file, or the file object itself
   :param h5path:
       String, or an iterable of strings to reach the parent of the leaves
   :param leaves: String, or iterable of strings with names of leaves to delete
   :return:
