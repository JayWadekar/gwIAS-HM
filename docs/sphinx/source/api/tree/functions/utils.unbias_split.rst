utils.unbias_split
==================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Function to compute split scores - expectations from total score

Signature
---------

.. code-block:: python

   def unbias_split(scores, vsq)

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
   * - ``vsq``
     - -
     - -
     - Array of length nchunk with fractional norm^2 of cosine waveforms, sums to one

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

   Function to compute split scores - expectations from total score
   :param scores: nchunk x nscore matrix with split scores
                  (can be vector for nscores=1)
   :param vsq:
       Array of length nchunk with fractional norm^2 of cosine waveforms,
       sums to one
   :return nchunk x nscore matrix with unbiased split scores
           (can be vector for nscores=1)
