triggers_single_detector_HM.TriggerList.from_gwosc
==================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Loads data from gwosc and makes a trigger file, records the event time as given by LIGO (different from the linear-free time) in t_event Make sure to pass an appropriate template_conf for the parameters of the event in load_kwargs If we are doing something custom (e.g., adding holes etc), save using to_json along with fname_preprocessing, and we can reload it later with the desired modifications passed to from_json

Signature
---------

.. code-block:: python

   def from_gwosc(cls, path = None, evname = 'GW150914', tgps = None, detector = 'H1', verbose = True, **load_kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``path``
     - -
     - None
     - Path to root directory where the file will be saved
   * - ``evname``
     - -
     - 'GW150914'
     - Event around which which we want the strain data
   * - ``tgps``
     - -
     - None
     - GPS time around which which we want the strain data
   * - ``detector``
     - -
     - 'H1'
     - Name of the detector for which we want the strain data
   * - ``verbose``
     - -
     - True
     - Print information
   * - ``\*\*load_kwargs``
     - -
     - -
     - Extra arguments to pass to the init function

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Instance of TriggerList

Docstring
---------

.. code-block:: text

   Loads data from gwosc and makes a trigger file, records the event
   time as given by LIGO (different from the linear-free time) in t_event
   Make sure to pass an appropriate template_conf for the parameters of
   the event in load_kwargs
   If we are doing something custom (e.g., adding holes etc), save using
   to_json along with fname_preprocessing, and we can reload it later
   with the desired modifications passed to from_json
   :param path: Path to root directory where the file will be saved
   :param evname: Event around which which we want the strain data
   :param tgps: GPS time around which which we want the strain data
   :param detector: Name of the detector for which we want the strain data
   :param verbose: Print information
   :param load_kwargs: Extra arguments to pass to the init function
   :return: Instance of TriggerList
