coherent_score_mz_fast.dt2key
=============================

Back to :doc:`Module page <../modules/coherent_score_mz_fast>`

Summary
-------

Jitable key for the dictionary from the time delays This is clunky but faster than the nice expression by O(1) when vectorized

Signature
---------

.. code-block:: python

   def dt2key(dt, dt_sinc = DEFAULT_DT, dt_max = DEFAULT_DT_MAX)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dt``
     - -
     - -
     - (ndelays = ndetectors-1) x nsamp array with time delays in milliseconds (should be 2d)
   * - ``dt_sinc``
     - -
     - DEFAULT_DT
     - Time resolution in milliseconds
   * - ``dt_max``
     - -
     - DEFAULT_DT_MAX
     - Maximum time delay in milliseconds

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_samp array of keys (always 1D)

Docstring
---------

.. code-block:: text

   Jitable key for the dictionary from the time delays
   This is clunky but faster than the nice expression by O(1) when vectorized
   :param dt:
       (ndelays = ndetectors-1) x nsamp array with time delays in milliseconds
       (should be 2d)
   :param dt_sinc: Time resolution in milliseconds
   :param dt_max: Maximum time delay in milliseconds
   :return: n_samp array of keys (always 1D)
