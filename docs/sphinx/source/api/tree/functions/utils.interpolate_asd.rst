utils.interpolate_asd
=====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

get log-log interpolant of ASD with old_f[0] >= 0,

Signature
---------

.. code-block:: python

   def interpolate_asd(old_f, old_asd, log_in = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``old_f``
     - -
     - -
     - (ordered) array of rfft frequencies >= 0 (in Hz) where ASD is saved
   * - ``old_asd``
     - -
     - -
     - len(old_f) array with ASD = sqrt(PSD)
   * - ``log_in``
     - -
     - False
     - bool indicating if old_asd is already a log (note old_f is never log)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Function that takes frequencies (Hz) & returns ASD at those frequencies

Docstring
---------

.. code-block:: text

   get log-log interpolant of ASD with old_f[0] >= 0,
   :param old_f: (ordered) array of rfft frequencies >= 0 (in Hz) where ASD is saved
   :param old_asd: len(old_f) array with ASD = sqrt(PSD)
   :param log_in: bool indicating if old_asd is already a log (note old_f is never log)
   :return: Function that takes frequencies (Hz) & returns ASD at those frequencies
