ranking_HM.Rank.compute_function_on_subbank_data
================================================

Back to :doc:`Class page <../classes/ranking_HM.Rank>`

Summary
-------

It can be slow to loop over the entries in scores_(non)vetoed_max, read them into memory, and compute something, in bulk. It is faster to dereference the hdf5 arrays in bg_fg_by_subbank once, compute what we need and read the values relevant to scores_(non)vetoed_max

Signature
---------

.. code-block:: python

   def compute_function_on_subbank_data(self, leafname, entry_idx, return_lists, func, *args, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``leafname``
     - -
     - -
     - Name of the hdf5 node to read (if known)
   * - ``entry_idx``
     - -
     - -
     - Index into an entry in bg_fg_by_subbank (used if leafname is None)
   * - ``return_lists``
     - -
     - -
     - Flag to return lists for cands_preveto_max and cands_postveto_max
   * - ``func``
     - -
     - -
     - Function to apply to the data
   * - ``\*args``
     - -
     - -
     - Arguments to pass to the function
   * - ``\*\*kwargs``
     - -
     - -
     - Keyword arguments to pass to the function

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. If not return_lists, a list of dicts with func evaluated for the entries in bg_fg_by_subbank 2. If return_lists, a list of dicts with func evaluated for the entries in bg_fg_by_subbank, and lists of func evaluated for the entries in cands_preveto_max, and cands_postveto_max

Docstring
---------

.. code-block:: text

   It can be slow to loop over the entries in scores_(non)vetoed_max, read
   them into memory, and compute something, in bulk. It is faster to
   dereference the hdf5 arrays in bg_fg_by_subbank once, compute what we
   need and read the values relevant to scores_(non)vetoed_max
   :param leafname: Name of the hdf5 node to read (if known)
   :param entry_idx:
       Index into an entry in bg_fg_by_subbank (used if leafname is None)
   :param return_lists:
       Flag to return lists for cands_preveto_max and cands_postveto_max
   :param func: Function to apply to the data
   :param args: Arguments to pass to the function
   :param kwargs: Keyword arguments to pass to the function
   :return:
       1. If not return_lists, a list of dicts with func evaluated for the
          entries in bg_fg_by_subbank
       2. If return_lists, a list of dicts with func evaluated for the
          entries in bg_fg_by_subbank, and lists of func evaluated for the
          entries in cands_preveto_max, and cands_postveto_max
