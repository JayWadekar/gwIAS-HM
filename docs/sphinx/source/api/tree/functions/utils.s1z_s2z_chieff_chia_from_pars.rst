utils.s1z_s2z_chieff_chia_from_pars
===================================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

\\\*NOTE\\\* this uses chia = (m1\\\*s1z - m2\\\*s2z) / (m1 + m2)

Signature
---------

.. code-block:: python

   def s1z_s2z_chieff_chia_from_pars(pe = True, **pars)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``pe``
     - -
     - True
     - -
   * - ``\*\*pars``
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
     - s1z, s2z, chieff, chia

Docstring
---------

.. code-block:: text

   *NOTE* this uses chia = (m1*s1z - m2*s2z) / (m1 + m2)
   :param **pars: pars must have at least two of mt, m1, m2,
                 and at least two of s1z (or chi1), s2z (or chi2), chia, chieff
   :return: s1z, s2z, chieff, chia
