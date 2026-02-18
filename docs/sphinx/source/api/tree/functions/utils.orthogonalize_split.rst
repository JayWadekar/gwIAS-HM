utils.orthogonalize_split
=========================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Function to compute orthogonalized scores to those in the L subset

Signature
---------

.. code-block:: python

   def orthogonalize_split(scores, proj_l, submask_l, submask_h = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``scores``
     - -
     - -
     - nchunk x nscore matrix with split scores (can be vector for nscores=1)
   * - ``proj_l``
     - -
     - -
     - Matrix to multiply scores in the L subset before subtracting from those in H to orthogonalize the latter
   * - ``submask_l``
     - -
     - -
     - Boolean mask into nchunk to select splits in L
   * - ``submask_h``
     - -
     - None
     - Boolean mask into nchunk to select splits in H (if not given, assume complement of submask_l)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - np.count_nonzero(submask_h) x nscore matrix with orthogonalized split scores (can be vector for nscores=1)

Docstring
---------

.. code-block:: text

   Function to compute orthogonalized scores to those in the L subset
   :param scores: nchunk x nscore matrix with split scores
                  (can be vector for nscores=1)
   :param proj_l:
       Matrix to multiply scores in the L subset before subtracting from those
       in H to orthogonalize the latter
   :param submask_l: Boolean mask into nchunk to select splits in L
   :param submask_h:
       Boolean mask into nchunk to select splits in H (if not given, assume
       complement of submask_l)
   :return:
       np.count_nonzero(submask_h) x nscore matrix with orthogonalized split
       scores (can be vector for nscores=1)
