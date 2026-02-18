utils.get_coincident_json_filelist
==================================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Assumes that the file names go like "Det-Detn-...." where Detn is the key

Signature
---------

.. code-block:: python

   def get_coincident_json_filelist(dir_name, enumerated_epochs = None, n_epochs = None, run = None, det1 = 'H1', det2 = 'L1')

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
   * - ``enumerated_epochs``
     - -
     - None
     - List of epochs, if needed
   * - ``n_epochs``
     - -
     - None
     - Number of epochs, if needed
   * - ``run``
     - -
     - None
     - String identifying the run
   * - ``det1``
     - -
     - 'H1'
     - Key identifying the first detector (e.g., "H1", "L1", "V1")
   * - ``det2``
     - -
     - 'L1'
     - Key identifying the second detector (e.g., "H1", "L1", "V1")

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. List of json files in the first detector (includes those with no coincidence with the second) 2. List of coincident json files in the second detector 3. Integer list of epochs corrresponding to 1 4. Integer list of epochs corrresponding to 2

Docstring
---------

.. code-block:: text

   Assumes that the file names go like "Det-Detn-...." where Detn is the key
   :param dir_name: Directory with json and trig files, created per subbank
   :param enumerated_epochs: List of epochs, if needed
   :param n_epochs: Number of epochs, if needed
   :param run: String identifying the run
   :param det1: Key identifying the first detector (e.g., "H1", "L1", "V1")
   :param det2: Key identifying the second detector (e.g., "H1", "L1", "V1")
   :return: 1. List of json files in the first detector
               (includes those with no coincidence with the second)
            2. List of coincident json files in the second detector
            3. Integer list of epochs corrresponding to 1
            4. Integer list of epochs corrresponding to 2
