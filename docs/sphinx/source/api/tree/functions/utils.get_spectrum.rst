utils.get_spectrum
==================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Function used to dechirp Gets a trigger in the format of trigger, bank_id

Signature
---------

.. code-block:: python

   def get_spectrum(trigger, det_ind = 1, N_freqs = 64, bns = True, use_HM = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - -
     - -
   * - ``det_ind``
     - -
     - 1
     - -
   * - ``N_freqs``
     - -
     - 64
     - -
   * - ``bns``
     - -
     - True
     - -
   * - ``use_HM``
     - -
     - False
     - -

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - -

Docstring
---------

.. code-block:: text

   Function used to dechirp
   Gets a trigger in the format of trigger, bank_id
   
   Returns:
       fs_spec,
       spectrum,              - power spectrum of the candidate
       power_spectrum_noise,  - average power spectrum of noise around the candidate
       asd,                   - The ASD at the time,
                                important if you want to compare to expectation
       dechirped_data         - Data around the event, dechirped such that the
                                candidate should contain all the power
                                in a single spectrum
