utils.incoherent_score
======================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def incoherent_score(triggers, no_sum = False, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``triggers``
     - -
     - -
     - n_cand x n_detector x row of processedclists (can be a 2d array if n_cand = 1)
   * - ``no_sum``
     - -
     - False
     - Flag to return individual scores
   * - ``\*\*kwargs``
     - -
     - -
     - -

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Vector of incoherent scores (scalar if n_cand = 1)

Docstring
---------

.. code-block:: text

   :param triggers: 
       n_cand x n_detector x row of processedclists
       (can be a 2d array if n_cand = 1)
   :param no_sum: Flag to return individual scores
   :return: Vector of incoherent scores (scalar if n_cand = 1)
