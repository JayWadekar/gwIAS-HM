coincidence_HM.flag_duplicates_per_group_pair
=============================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Picks a subset of pair_trigger_array such that each bucket of group_size in H1 and L1 has a unique \\\`coincident' candidate. If a veto_mask is provided, defaults to preferentially keep in vetoed elements Note: The order of the input and output pair trigger array isn't identical in the general case

Signature
---------

.. code-block:: python

   def flag_duplicates_per_group_pair(pair_trigger_array, group_size, c0_pos, origin = 0, veto_mask = None, extra_arrays = None, remove = False, score_func = utils.incoherent_score, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``pair_trigger_array``
     - -
     - -
     - n_trigger x 2 x len(processedclist) array with H1 and L1 triggers
   * - ``group_size``
     - -
     - -
     - Bucket size (s)
   * - ``c0_pos``
     - -
     - -
     - Index of c0 in the processedclists, used if remove == False
   * - ``origin``
     - -
     - 0
     - Origin for bucketing (s)
   * - ``veto_mask``
     - -
     - None
     - If known, boolean mask of len(pair_trigger_array) with zeros at triggers that failed vetoes
   * - ``extra_arrays``
     - -
     - None
     - If required, a list of arrays, each of size n_trigger, to dice up along with pair_trigger_array
   * - ``remove``
     - -
     - False
     - If True, we remove the duplicates. Else we return a boolean mask that marks the duplicates with zeros
   * - ``score_func``
     - -
     - utils.incoherent_score
     - Function that takes in a pair of triggers and returns a scalar
   * - ``\*\*kwargs``
     - -
     - -
     - Extra arguments to score_func

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. n_trigger_new x 2 x len(processedclist) array with subset of maximized H1 and L1 triggers 2. If veto_mask is provided, subset of veto_mask of size n_trigger_new (if provided in the first place) 3. If extra_arrays are provided, list with their subsets 4. if remove is False, n_trigger_new = n_trigger, and it returns a n_trigger x 2 boolean mask with zeros at the eliminated triggers in each detector

Docstring
---------

.. code-block:: text

   Picks a subset of pair_trigger_array such that each bucket of group_size
   in H1 and L1 has a unique `coincident' candidate. If a veto_mask is
   provided, defaults to preferentially keep in vetoed elements
   Note: The order of the input and output pair trigger array isn't identical
   in the general case
   :param pair_trigger_array:
       n_trigger x 2 x len(processedclist) array with H1 and L1 triggers
   :param group_size: Bucket size (s)
   :param origin: Origin for bucketing (s)
   :param veto_mask:
       If known, boolean mask of len(pair_trigger_array) with zeros at triggers
       that failed vetoes
   :param extra_arrays:
       If required, a list of arrays, each of size n_trigger, to dice up
       along with pair_trigger_array
   :param remove:
       If True, we remove the duplicates. Else we return a boolean mask that
       marks the duplicates with zeros
   :param c0_pos: Index of c0 in the processedclists, used if remove == False
   :param score_func:
       Function that takes in a pair of triggers and returns a scalar
   :param kwargs: Extra arguments to score_func
   :return: 1. n_trigger_new x 2 x len(processedclist) array with
               subset of maximized H1 and L1 triggers
            2. If veto_mask is provided, subset of veto_mask of
               size n_trigger_new (if provided in the first place)
            3. If extra_arrays are provided, list with their subsets
            4. if remove is False, n_trigger_new = n_trigger, and it returns
               a n_trigger x 2 boolean mask with zeros at the eliminated
               triggers in each detector
