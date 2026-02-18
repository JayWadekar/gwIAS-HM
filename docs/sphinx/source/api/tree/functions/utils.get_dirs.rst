utils.get_dirs
==============

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Gives a list of length n_runs with each entry being a dictionary of dictionaries with the outer dictionary having chirp mass ids as keys and the inner dictionary having subbank ids as keys and subdirectory names as values. Figures out the number of subbanks and chirp mass ids by itself

Signature
---------

.. code-block:: python

   def get_dirs(dtype = 'trigs', vers_suffix = '', source = 'BBH', runs = ('O2',))

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dtype``
     - -
     - 'trigs'
     - Type of directory (trigs, cand, stats)
   * - ``vers_suffix``
     - -
     - ''
     - Suffix after the directory name if needed (e.g., for candidates version 4, we used 'cand4' in the past)
   * - ``source``
     - -
     - 'BBH'
     - Source type (BBH, NSBH, BNS)
   * - ``runs``
     - -
     - ('O2',)
     - List of run names

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - list of n_runs dictionaries with chirp mass ids as keys and list of subbank directories as values

Docstring
---------

.. code-block:: text

   Gives a list of length n_runs with each entry being a dictionary of
   dictionaries with the outer dictionary having chirp mass ids as keys and
   the inner dictionary having subbank ids as keys and subdirectory names as
   values. Figures out the number of subbanks and chirp mass ids by itself
   :param dtype: Type of directory (trigs, cand, stats)
   :param vers_suffix:
       Suffix after the directory name if needed
       (e.g., for candidates version 4, we used 'cand4' in the past)
   :param source: Source type (BBH, NSBH, BNS)
   :param runs: List of run names
   :return:
       list of n_runs dictionaries with chirp mass ids as keys and list of
       subbank directories as values
