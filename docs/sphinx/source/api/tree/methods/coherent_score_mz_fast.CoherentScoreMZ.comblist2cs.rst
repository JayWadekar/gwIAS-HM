coherent_score_mz_fast.CoherentScoreMZ.comblist2cs
==================================================

Back to :doc:`Class page <../classes/coherent_score_mz_fast.CoherentScoreMZ>`

Summary
-------

Takes the return value of trigger2comblist and returns the coherent score by calling the jitted function

Signature
---------

.. code-block:: python

   def comblist2cs(self, timeseries, offsets, nfacs, gtype = 1, nsamples = 10000, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``timeseries``
     - -
     - -
     - Tuple of length n_detectors with n_samp x 3 array with Re(z), Im(z)
   * - ``offsets``
     - -
     - -
     - Array of length n_detector -1 with shifts to apply to the timeseries > H1 to bring them to zero lag
   * - ``nfacs``
     - -
     - -
     - Array of length n_detector with detector sensitivities
   * - ``gtype``
     - -
     - 1
     - If 0, turns off the integration over distance
   * - ``nsamples``
     - -
     - 10000
     - Number of samples to use in the monte carlo
   * - ``\*\*kwargs``
     - -
     - -
     - Generic variable to capture any extra arguments

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Coherent score, defined as 2 \\\* log (p(H_1) / p(H_0))

Docstring
---------

.. code-block:: text

   Takes the return value of trigger2comblist and returns the coherent
   score by calling the jitted function
   :param timeseries:
       Tuple of length n_detectors with n_samp x 3 array with Re(z), Im(z)
   :param offsets:
       Array of length n_detector -1 with shifts to apply to the
       timeseries > H1 to bring them to zero lag
   :param nfacs: Array of length n_detector with detector sensitivities
   :param gtype: If 0, turns off the integration over distance
   :param nsamples: Number of samples to use in the monte carlo
   :param kwargs: Generic variable to capture any extra arguments
   :return: Coherent score, defined as 2 * log (p(H_1) / p(H_0))
