ranking_HM.check_and_fix_bg_fg_lists
====================================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

Our injection codes populate bg_fg_max, but not bg_fg_by_subbank. We also want to save indices into bg_fg_by_subbank in bg_fg_max to avoid duplication in memory, but old codes might not have saved it in this format. This function fixes both issues. Run this before scoring, and re-score after.

Signature
---------

.. code-block:: python

   def check_and_fix_bg_fg_lists(bg_fg_max, bg_fg_by_subbank, extra_array_names, fobj = None, raise_error = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``bg_fg_max``
     - -
     - -
     - List of four lists with entries like bg_fg_max
   * - ``bg_fg_by_subbank``
     - -
     - -
     - List of four dicts with entries like bg_fg_by_subbank
   * - ``extra_array_names``
     - -
     - -
     - List of names of the extra arrays in bg_fg_max
   * - ``fobj``
     - -
     - None
     - If bg_by_subbank is read from a hdf5 file, the File object (must be writeable)
   * - ``raise_error``
     - -
     - False
     - Flag to raise an error if the number of triggers don't match, just use as a check

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Fixes bg_fg_max and bg_fg_by_subbank in place, and also fixes the data inside fobj

Docstring
---------

.. code-block:: text

   Our injection codes populate bg_fg_max, but not bg_fg_by_subbank.
   We also want to save indices into bg_fg_by_subbank in bg_fg_max to avoid
   duplication in memory, but old codes might not have saved it in this
   format. This function fixes both issues. Run this before scoring, and
   re-score after.
   :param bg_fg_max: List of four lists with entries like bg_fg_max
   :param bg_fg_by_subbank:
       List of four dicts with entries like bg_fg_by_subbank
   :param extra_array_names: List of names of the extra arrays in bg_fg_max
   :param fobj:
       If bg_by_subbank is read from a hdf5 file, the File object
       (must be writeable)
   :param raise_error:
       Flag to raise an error if the number of triggers don't match, just use
       as a check
   :return:
       Fixes bg_fg_max and bg_fg_by_subbank in place, and also fixes the data
       inside fobj
