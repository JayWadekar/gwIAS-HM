coincidence_HM.get_best_candidate_segments
==========================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Returns best trigger pair given two lists of friends. Doesn't impose a time-delay constraint, assumes that time-delay is either irrelevant (bg), or taken care of by using the appropriate time_shift_tol

Signature
---------

.. code-block:: python

   def get_best_candidate_segments(friends_arr1, friends_arr2, c0_pos, score_func = utils.incoherent_score, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``friends_arr1``
     - -
     - -
     - processedclist of triggers
   * - ``friends_arr2``
     - -
     - -
     - processedclist of triggers
   * - ``c0_pos``
     - -
     - -
     - Index of c0 in processedclists
   * - ``score_func``
     - -
     - utils.incoherent_score
     - Function that accepts coincident trigger(s) and returns score(s) (in the future maybe split into incoherent and coherent score funcs)
   * - ``\*\*kwargs``
     - -
     - -
     - Dictionary with extra parameters for score_func

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - tuple of len(2) with H1 trigger and L1 trigger

Docstring
---------

.. code-block:: text

   Returns best trigger pair given two lists of friends. Doesn't impose a
   time-delay constraint, assumes that time-delay is either irrelevant (bg),
   or taken care of by using the appropriate time_shift_tol
   :param friends_arr1: processedclist of triggers
   :param friends_arr2: processedclist of triggers
   :param c0_pos: Index of c0 in processedclists
   :param score_func:
       Function that accepts coincident trigger(s) and returns score(s)
       (in the future maybe split into incoherent and coherent score funcs)
   :param kwargs: Dictionary with extra parameters for score_func
   :return: tuple of len(2) with H1 trigger and L1 trigger
