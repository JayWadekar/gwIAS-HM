triggers_single_detector_HM.TriggerList.from_json
=================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Instantiates class from json file created by previous run/gen_triggers Load using load_data=True and do_signal_processing=False to access the data without any glitch rejection

Signature
---------

.. code-block:: python

   def from_json(cls, config_fname, load_trigs = True, load_preproc = True, do_ffts = 2, load_data = False, do_signal_processing = True, return_instance = True, load_injected_waveforms = False, **kwargs)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``config_fname``
     - -
     - -
     - Absolute path of json file
   * - ``load_trigs``
     - -
     - True
     - Flag indicating whether to load triggers
   * - ``load_preproc``
     - -
     - True
     - Flag indicating whether to load preprocessing (if False, doesn't load the npz file)
   * - ``do_ffts``
     - -
     - 2
     - Flag indicating whether to define templatebank and chunked FFTs (only used if load_preproc or load_data is True) 0: No FFTs 1: Sets up the template bank 2. Sets up chunked data If boolean, False = 0, True = 2
   * - ``load_data``
     - -
     - False
     - Set flag to False in order to avoid time intensive operations by not touching the data. If True, does preprocessing even if file exists
   * - ``do_signal_processing``
     - -
     - True
     - Flag indicating whether to do signal processing to identify glitches and fill holes. Only relevant if load_data is True
   * - ``return_instance``
     - -
     - True
     - -
   * - ``load_injected_waveforms``
     - -
     - False
     - Inidcates whether injected waveforms need to be loaded from a separate file with filename(s) injection_args['wf_filename'].
   * - ``\*\*kwargs``
     - -
     - -
     - Override any of the other input parameters to init

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

   Instantiates class from json file created by previous run/gen_triggers
   Load using load_data=True and do_signal_processing=False to access the
   data without any glitch rejection
   :param config_fname: Absolute path of json file
   :param load_trigs: Flag indicating whether to load triggers
   :param load_preproc:
       Flag indicating whether to load preprocessing (if False, doesn't
       load the npz file)
   :param do_ffts:
       Flag indicating whether to define templatebank and chunked FFTs
       (only used if load_preproc or load_data is True)
       0: No FFTs
       1: Sets up the template bank
       2. Sets up chunked data
       If boolean, False = 0, True = 2
   :param load_data:
       Set flag to False in order to avoid time intensive operations by not
       touching the data. If True, does preprocessing even if file exists
   :param do_signal_processing:
       Flag indicating whether to do signal processing to identify glitches
       and fill holes. Only relevant if load_data is True
   :param load_injected_waveforms:
       Inidcates whether injected waveforms need to be loaded from a separate
       file with filename(s) injection_args['wf_filename'].
   :param kwargs: Override any of the other input parameters to init
   :return: Instance of TriggerList
