utils.submask
=============

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def submask(bigmask, *indarrays)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``bigmask``
     - -
     - -
     - Boolean mask that picks out a subset of \\\`valid' elements within a set
   * - ``\*indarrays``
     - -
     - -
     - Each indarray is either a list of indices or a Boolean mask, into the parent set of bigmask, to be retained, subject to validity

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Boolean mask (of length np.count_nonzero(bigmask)) into the set of \\\`valid' elements that picks out elements indexed by the union of the indarrays

Docstring
---------

.. code-block:: text

   :param bigmask:
       Boolean mask that picks out a subset of `valid' elements within a set
   :param indarrays:
       Each indarray is either a list of indices or a Boolean mask, into the
       parent set of bigmask, to be retained, subject to validity
   :return: Boolean mask (of length np.count_nonzero(bigmask)) into the set of
   `valid' elements that picks out elements indexed by the union of the
   indarrays
