utils.get_detector_fnames
=========================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def get_detector_fnames(t, chirp_mass_id = 0, subbank = None, run = None, source = 'BBH', use_HM = False, dname = None, detectors = ('H1', 'L1', 'V1'), return_only_existing = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``t``
     - -
     - -
     - GPS time
   * - ``chirp_mass_id``
     - -
     - 0
     - Chirp mass bank id
   * - ``subbank``
     - -
     - None
     - Subbank id, if None, returns all subbanks
   * - ``run``
     - -
     - None
     - Run name, if None, figures it out from t and use_HM
   * - ``source``
     - -
     - 'BBH'
     - Source type (BBH, NSBH, BNS)
   * - ``use_HM``
     - -
     - False
     - If True, uses HM run names
   * - ``dname``
     - -
     - None
     - If not None, returns the json files in this directory
   * - ``detectors``
     - -
     - ('H1', 'L1', 'V1')
     - List of detectors to return files for
   * - ``return_only_existing``
     - -
     - True
     - If True, returns only existing files and None for missing ones

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - An n_detector array of json file names for the TriggerLists at time t

Docstring
---------

.. code-block:: text

   :param t: GPS time
   :param chirp_mass_id: Chirp mass bank id
   :param subbank: Subbank id, if None, returns all subbanks
   :param run: Run name, if None, figures it out from t and use_HM
   :param source: Source type (BBH, NSBH, BNS)
   :param use_HM: If True, uses HM run names
   :param dname: If not None, returns the json files in this directory
   :param detectors: List of detectors to return files for
   :param return_only_existing:
       If True, returns only existing files and None for missing ones
   :return:
       An n_detector array of json file names for the TriggerLists at time t
