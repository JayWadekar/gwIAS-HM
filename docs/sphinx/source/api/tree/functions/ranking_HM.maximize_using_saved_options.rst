ranking_HM.maximize_using_saved_options
=======================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Convenience function to use saved options to maximize_over_banks

Signature
---------

.. code-block:: python

   def maximize_using_saved_options(list_of_rank_objs, maxopts_filepath, keys_to_pop = None, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``list_of_rank_objs``
     - -
     - -
     - List of Rank objects
   * - ``maxopts_filepath``
     - -
     - -
     - Path to the hdf5 file with the maximization options
   * - ``keys_to_pop``
     - -
     - None
     - Keys to pop from the arguments, if needed (adding this in case we lose some options in the future)
   * - ``\*\*kwargs``
     - -
     - -
     - Any extra arguments we want to override or pass to the maximization

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Performs the maximization over banks

Docstring
---------

.. code-block:: text

   Convenience function to use saved options to maximize_over_banks
   :param list_of_rank_objs: List of Rank objects
   :param maxopts_filepath: Path to the hdf5 file with the maximization options
   :param keys_to_pop:
       Keys to pop from the arguments, if needed
       (adding this in case we lose some options in the future)
   :param kwargs:
       Any extra arguments we want to override or pass to the maximization
   :return: Performs the maximization over banks
