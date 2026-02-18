utils.standardize_run_name
==========================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Maintain consistency in run names, mainly used to confirm with the GWOSC naming conventions

Signature
---------

.. code-block:: python

   def standardize_run_name(run_name)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``run_name``
     - -
     - -
     - Run name to be standardized

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

   Maintain consistency in run names, mainly used to confirm with the GWOSC
   naming conventions
   :param run_name: Run name to be standardized
   :return Standardized run name, with leading hm_ removed and capitalized
