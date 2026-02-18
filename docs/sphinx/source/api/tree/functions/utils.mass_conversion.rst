utils.mass_conversion
=====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Computes all conversions given a subset of parameters describing the masses

Signature
---------

.. code-block:: python

   def mass_conversion(**dic)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``\*\*dic``
     - -
     - -
     - Dictionary with known subset of mc, mt, m1, m2, q, eta (all can be scalars or numpy arrays)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - dic with all values solved for

Docstring
---------

.. code-block:: text

   Computes all conversions given a subset of parameters describing the masses
   :param dic:
       Dictionary with known subset of mc, mt, m1, m2, q, eta
       (all can be scalars or numpy arrays)
   :return: dic with all values solved for
