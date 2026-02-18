utils.threshold_rv
==================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def threshold_rv(dist, nsamp, *dist_args, nfire = params.NPERFILE, onesided = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dist``
     - -
     - -
     - Distribution in scipy.stats
   * - ``nsamp``
     - -
     - -
     - Number of samples
   * - ``\*dist_args``
     - -
     - -
     - Extra arguments for distribution
   * - ``nfire``
     - -
     - params.NPERFILE
     - Number of times noise should cross the threshold in samples
   * - ``onesided``
     - -
     - True
     - False if we want lower threshold too

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Threshold for random variable that is exceeded nfire times in nsamp samples on average (upper and lower thresholds if onesided=False)

Docstring
---------

.. code-block:: text

   :param dist: Distribution in scipy.stats
   :param nsamp: Number of samples
   :param dist_args: Extra arguments for distribution
   :param nfire: Number of times noise should cross the threshold in samples
   :param onesided: False if we want lower threshold too
   :return: Threshold for random variable that is exceeded nfire times in
            nsamp samples on average (upper and lower thresholds if
            onesided=False)
