coherent_score_mz_fast.marg_lk
==============================

Back to :doc:`Module page <../modules/coherent_score_mz_fast>`

Summary
-------

Computes likelihood marginalized over distance and orbital phase

Signature
---------

.. code-block:: python

   def marg_lk(zz, tt, gtype = 1, nsamp = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``zz``
     - -
     - -
     - nsamp x n_detector array with rows having complex overlaps for each detector
   * - ``tt``
     - -
     - -
     - nsamp x n_detector array with predicted overlaps in each detector for fiducial orbital phase and distance (upto an arbitrary scaling factor)
   * - ``gtype``
     - -
     - 1
     - passed to function to compute marginalization over distance and phase, determines which one to run
   * - ``nsamp``
     - -
     - None
     - Pass to only sum over part of the arrays

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - nsamp array of likelihoods marginalized over the orbital phase, and distance if needed (always 1D)

Docstring
---------

.. code-block:: text

   Computes likelihood marginalized over distance and orbital phase
   :param zz:
       nsamp x n_detector array with rows having complex overlaps for
       each detector
   :param tt: nsamp x n_detector array with predicted overlaps in each
       detector for fiducial orbital phase and distance
       (upto an arbitrary scaling factor)
   :param gtype: passed to function to compute marginalization over distance
       and phase, determines which one to run
   :param nsamp: Pass to only sum over part of the arrays
   :returns:
       nsamp array of likelihoods marginalized over the orbital phase, and
       distance if needed (always 1D)
