coincidence_HM.group_by_id
==========================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Groups the triggers in a clist into sublists with common calphas

Signature
---------

.. code-block:: python

   def group_by_id(clist, c0_pos, ncalpha = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``clist``
     - -
     - -
     - Processedclist
   * - ``c0_pos``
     - -
     - -
     - Index of c0
   * - ``ncalpha``
     - -
     - None
     - Use only up to ncalpha coefficients if needed, useful for making heatmaps

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. List of unique hashes (template ids) of templates in clist 2. List of sublists of the clist with each sublist having triggers with a common template id

Docstring
---------

.. code-block:: text

   Groups the triggers in a clist into sublists with common calphas
   :param clist: Processedclist
   :param c0_pos: Index of c0
   :param ncalpha:
       Use only up to ncalpha coefficients if needed, useful for making
       heatmaps
   :return: 1. List of unique hashes (template ids) of templates in clist
            2. List of sublists of the clist with each sublist having triggers
               with a common template id
