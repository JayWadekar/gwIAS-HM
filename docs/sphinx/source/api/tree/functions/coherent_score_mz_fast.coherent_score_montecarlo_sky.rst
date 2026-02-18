coherent_score_mz_fast.coherent_score_montecarlo_sky
====================================================

Back to :doc:`Module page <../modules/coherent_score_mz_fast>`

Summary
-------

# TODO: Avoid tuples, make varargs? Evaluates the coherent score integral by montecarlo sampling all relevant variables

Signature
---------

.. code-block:: python

   def coherent_score_montecarlo_sky(timeseries, offsets, nfacs, dt_dict_keys, dt_dict_items, responses, t3norm, musamps = None, psisamps = None, gtype = 1, dt_sinc = DEFAULT_DT, dt_max = DEFAULT_DT_MAX, nsamples = 10000)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``timeseries``
     - -
     - -
     - Tuple with n_samp x 3 arrays with times, Re(z), Im(z) in each detector
   * - ``offsets``
     - -
     - -
     - (n_det - 1) array with offsets to add to the detectors > first one (e.g., H1 for H1, L1) to bring them to zero lag
   * - ``nfacs``
     - -
     - -
     - n_det array of instantaneous sensitivities in each detector (normfac/psd_drift x hole_correction)
   * - ``dt_dict_keys``
     - -
     - -
     - Keys to dictionary computed using the delays in each dt_tuple, sorted
   * - ``dt_dict_items``
     - -
     - -
     - Values in dictionary, tuple of n_sky x 2 arrays with indices into ras, decs for each allowed dt tuple
   * - ``responses``
     - -
     - -
     - n_ra x n_dec x n_detector x 2 array with detector responses
   * - ``t3norm``
     - -
     - -
     - normalization constant such that prior integrated over the sky = 1
   * - ``musamps``
     - -
     - None
     - If available, array with samples of mu (cos inclination)
   * - ``psisamps``
     - -
     - None
     - If available, array with samples of psi (pol angle)
   * - ``gtype``
     - -
     - 1
     - 0/1 to not marginalize/marginalize over distance
   * - ``dt_sinc``
     - -
     - DEFAULT_DT
     - Time resolution for the dictionary (ms)
   * - ``dt_max``
     - -
     - DEFAULT_DT_MAX
     - Rough upper bound on the individual delays (ms)
   * - ``nsamples``
     - -
     - 10000
     - Number of samples for montecarlo evaluation

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Montecarlo evaluation of complete coherent score (including the incoherent part)

Docstring
---------

.. code-block:: text

   # TODO: Avoid tuples, make varargs?
   Evaluates the coherent score integral by montecarlo sampling all
   relevant variables
   :param timeseries:
       Tuple with n_samp x 3 arrays with times, Re(z), Im(z) in each detector
   :param offsets:
       (n_det - 1) array with offsets to add to the detectors > first one
       (e.g., H1 for H1, L1) to bring them to zero lag
   :param nfacs:
       n_det array of instantaneous sensitivities in each detector
       (normfac/psd_drift x hole_correction)
   :param dt_dict_keys:
       Keys to dictionary computed using the delays in each dt_tuple, sorted
   :param dt_dict_items:
       Values in dictionary, tuple of n_sky x 2 arrays with indices into
       ras, decs for each allowed dt tuple
   :param responses:
       n_ra x n_dec x n_detector x 2 array with detector responses
   :param t3norm:
       normalization constant such that prior integrated over the sky = 1
   :param musamps: If available, array with samples of mu (cos inclination)
   :param psisamps: If available, array with samples of psi (pol angle)
   :param gtype: 0/1 to not marginalize/marginalize over distance
   :param dt_sinc: Time resolution for the dictionary (ms)
   :param dt_max: Rough upper bound on the individual delays (ms)
   :param nsamples: Number of samples for montecarlo evaluation
   :returns:
       Montecarlo evaluation of complete coherent score
       (including the incoherent part)
