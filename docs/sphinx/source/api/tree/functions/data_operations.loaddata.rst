data_operations.loaddata
========================

Back to :doc:`Module page <../modules/data_operations>`

Summary
-------

Load data Warning: 1. Assumes that the quality flags are sampled at 1 Hz 2. Assumes that the sampling rate of data is an integer # TODO: If mask is too intermittent, make it flat since it is unreliable...

Signature
---------

.. code-block:: python

   def loaddata(fname = None, left_fname = None, right_fname = None, quality_flags = ('DEFAULT', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3'), chunktime_psd = params.DEF_CHUNKTIME_PSD, wf_duration_est = params.MIN_WFDURATION, edgepadding = True, fmax = params.FMAX_OVERLAP, **load_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fname``
     - -
     - None
     - Filename
   * - ``left_fname``
     - -
     - None
     - File with strain data on the left, if known
   * - ``right_fname``
     - -
     - None
     - File with strain data on the right, if known
   * - ``quality_flags``
     - -
     - ('DEFAULT', 'CBC_CAT1', 'CBC_CAT2', 'CBC_CAT3')
     - Tuple of LIGO data quality flags to apply cuts on Check out https://losc.ligo.org/techdetails/
   * - ``chunktime_psd``
     - -
     - params.DEF_CHUNKTIME_PSD
     - Length of time chunk for measuring PSD (s)
   * - ``wf_duration_est``
     - -
     - params.MIN_WFDURATION
     - Estimated waveform duration (used to figure out how much to pull in)
   * - ``edgepadding``
     - -
     - True
     - Flag indicating whether to pad data with zeros that we will fill later
   * - ``fmax``
     - -
     - params.FMAX_OVERLAP
     - Maximum frequency involved in the analysis, used to estimate buffer indices
   * - ``\*\*load_kwargs``
     - -
     - -
     - Dictionary with details to load from GPS times instead of files. Should have the following keys: 'run': O1/O2/O3a 'IFO': H1/L1/V1 'tstart': Starting time 'tstop': Stopping time No loading extra seconds

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. Times 2. Strains 3. Boolean mask with quality flags applied 4. Loaded channel_dict with all flags (at 1 Hz resolution) 5. Time at the left edge of central file

Docstring
---------

.. code-block:: text

   Load data
   Warning:
   1. Assumes that the quality flags are sampled at 1 Hz
   2. Assumes that the sampling rate of data is an integer
   # TODO: If mask is too intermittent, make it flat since it is unreliable...
   :param fname: Filename
   :param left_fname: File with strain data on the left, if known
   :param right_fname: File with strain data on the right, if known
   :param quality_flags:
       Tuple of LIGO data quality flags to apply cuts on
       Check out https://losc.ligo.org/techdetails/
   :param chunktime_psd: Length of time chunk for measuring PSD (s)
   :param wf_duration_est:
       Estimated waveform duration (used to figure out how much to pull in)
   :param edgepadding:
       Flag indicating whether to pad data with zeros that we will fill later
   :param fmax:
       Maximum frequency involved in the analysis, used to estimate
       buffer indices
   :param load_kwargs:
       Dictionary with details to load from GPS times instead of files.
       Should have the following keys:
       'run': O1/O2/O3a
       'IFO': H1/L1/V1
       'tstart': Starting time
       'tstop': Stopping time
       No loading extra seconds
   :return: 1. Times
            2. Strains
            3. Boolean mask with quality flags applied
            4. Loaded channel_dict with all flags (at 1 Hz resolution)
            5. Time at the left edge of central file
