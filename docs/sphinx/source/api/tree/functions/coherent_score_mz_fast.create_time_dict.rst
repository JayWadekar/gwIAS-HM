coherent_score_mz_fast.create_time_dict
=======================================

Back to :doc:`Module page <../modules/coherent_score_mz_fast>`

Summary
-------

Creates dictionary indexed by time, giving RA-Dec pairs for montecarlo integration

Signature
---------

.. code-block:: python

   def create_time_dict(nra, ndec, detectors, gps_time = 1136574828.0, dt_sinc = DEFAULT_DT, dt_max = DEFAULT_DT_MAX)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``nra``
     - -
     - -
     - number of ra points in grid
   * - ``ndec``
     - -
     - -
     - number of declinations in grid
   * - ``detectors``
     - -
     - -
     - List of detectors, each with a location and response as given by LAL
   * - ``gps_time``
     - -
     - 1136574828.0
     - Reference GPS time to generate the dictionary for (arbitrary for our use case)
   * - ``dt_sinc``
     - -
     - DEFAULT_DT
     - size of the time binning used in ms; it must coincide with the time separation of samples in the overlap if this dictionary will be used for doing the marginalization
   * - ``dt_max``
     - -
     - DEFAULT_DT_MAX
     - Rough upper bound on the individual delays

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 0. Dictionary indexed by the dts key, returning n_sky x 2 array with indices into ras and decs for each allowed dt tuple 1. List of ras 2. List of decs 3. n_ra x n_dec x n_detector x 2 array with responses 4. n_ra x n_dec x (n_detector - 1) array with delta ts 5. n_ra x n_dec x (n_detector - 1) array with delta phis 6. n_ra x n_dec array with network sensitivity (sum_{det, pol} resp^2)

Docstring
---------

.. code-block:: text

   Creates dictionary indexed by time, giving RA-Dec pairs for montecarlo
   integration
   :param nra: number of ra points in grid
   :param ndec: number of declinations in grid
   :param detectors:
       List of detectors, each with a location and response as given by LAL
   :param gps_time: Reference GPS time to generate the dictionary for
       (arbitrary for our use case)
   :param dt_sinc:
       size of the time binning used in ms; it must coincide with the time
       separation of samples in the overlap if this dictionary will be used
       for doing the marginalization
   :param dt_max: Rough upper bound on the individual delays
   :return:
       0. Dictionary indexed by the dts key, returning n_sky x 2 array with
          indices into ras and decs for each allowed dt tuple
       1. List of ras
       2. List of decs
       3. n_ra x n_dec x n_detector x 2 array with responses
       4. n_ra x n_dec x (n_detector - 1) array with delta ts
       5. n_ra x n_dec x (n_detector - 1) array with delta phis
       6. n_ra x n_dec array with network sensitivity (sum_{det, pol} resp^2)
