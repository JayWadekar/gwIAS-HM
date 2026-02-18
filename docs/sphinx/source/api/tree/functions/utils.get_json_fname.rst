utils.get_json_fname
====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

TODO: What if we choose a different fmax? Assumes that the file names go like "Det-Detn-...." where Detn is the detector key

Signature
---------

.. code-block:: python

   def get_json_fname(dir_name, epoch, detector, run = 'O3a')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``dir_name``
     - -
     - -
     - Directory with json and trig files, created per subbank
   * - ``epoch``
     - -
     - -
     - Integer with epoch (t0)
   * - ``detector``
     - -
     - -
     - Key identifying the detector (e.g., "H1", "L1", "V1")
   * - ``run``
     - -
     - 'O3a'
     - String identifying the run

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Absolute json fname

Docstring
---------

.. code-block:: text

   TODO: What if we choose a different fmax?
   Assumes that the file names go like "Det-Detn-...." where Detn is the
   detector key
   :param dir_name: Directory with json and trig files, created per subbank
   :param epoch: Integer with epoch (t0)
   :param detector: Key identifying the detector (e.g., "H1", "L1", "V1")
   :param run: String identifying the run
   :return: Absolute json fname
