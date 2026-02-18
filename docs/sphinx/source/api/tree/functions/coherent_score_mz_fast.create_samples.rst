coherent_score_mz_fast.create_samples
=====================================

Back to :doc:`Module page <../modules/coherent_score_mz_fast>`

Summary
-------

Create samples and save to file

Signature
---------

.. code-block:: python

   def create_samples(fname, nra = 100, ndec = 100, detnames = ('H1', 'L1'), gps_time = 1136574828.0, dt_sinc = DEFAULT_DT, dt_max = DEFAULT_DT_MAX, nsamples = 100000)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fname``
     - -
     - -
     - File name of archive to create
   * - ``nra``
     - -
     - 100
     - Number of right ascensions
   * - ``ndec``
     - -
     - 100
     - Number of declinations
   * - ``detnames``
     - -
     - ('H1', 'L1')
     - Names of detectors for the structure
   * - ``gps_time``
     - -
     - 1136574828.0
     - Fiducial GPS time for mapping between the RA-DEC and time delays (arbitrary for typical usage)
   * - ``dt_sinc``
     - -
     - DEFAULT_DT
     - Time resolution for marginalization
   * - ``dt_max``
     - -
     - DEFAULT_DT_MAX
     - Rough upper bound on the individual delays
   * - ``nsamples``
     - -
     - 100000
     - Number of random samples of the inclination and the polarization

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

   Create samples and save to file
   :param fname: File name of archive to create
   :param nra: Number of right ascensions
   :param ndec: Number of declinations
   :param detnames: Names of detectors for the structure
   :param gps_time:
       Fiducial GPS time for mapping between the RA-DEC and time delays
       (arbitrary for typical usage)
   :param dt_sinc: Time resolution for marginalization
   :param dt_max: Rough upper bound on the individual delays
   :param nsamples:
       Number of random samples of the inclination and the polarization
   :return:
