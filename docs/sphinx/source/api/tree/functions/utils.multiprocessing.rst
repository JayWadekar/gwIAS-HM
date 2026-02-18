utils.multiprocessing
=====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

A shorthand function to apply multiprocessing in a one liner

Signature
---------

.. code-block:: python

   def multiprocessing(func, input_list, num_processors)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``func``
     - -
     - -
     - Function to apply. Callable with single objects in input list
   * - ``input_list``
     - -
     - -
     - inputs for the function.
   * - ``num_processors``
     - -
     - -
     - number of processors to use. If 1, will apply locally in the shell.

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - the list [func(inp) for inp in input_list]

Docstring
---------

.. code-block:: text

   A shorthand function to apply multiprocessing in a one liner
   :param func: Function to apply. Callable with single objects in input list
   :param input_list: inputs for the function.
   :param num_processors: number of processors to use. If 1, will apply locally in the shell.
   :return: the list [func(inp) for inp in input_list]
