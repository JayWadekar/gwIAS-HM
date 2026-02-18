coherent_score_mz_fast.phase_lags
=================================

Back to :doc:`Module page <../modules/coherent_score_mz_fast>`

Summary
-------

Computes phase lags between detectors, referred to the first (beyond 2 delays for most, and 3 delays for all, they're overspecified, but that's OK)

Signature
---------

.. code-block:: python

   def phase_lags(ra, dec, detectors, gps_time)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``ra``
     - -
     - -
     - -
   * - ``dec``
     - -
     - -
     - -
   * - ``detectors``
     - -
     - -
     - -
   * - ``gps_time``
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
     - -

Docstring
---------

.. code-block:: text

   Computes phase lags between detectors, referred to the first
   (beyond 2 delays for most, and 3 delays for all, they're overspecified,
   but that's OK)
