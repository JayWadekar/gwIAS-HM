triggers_single_detector_HM.TriggerList.gen_overlaps
====================================================

Back to :doc:`Class page <../classes/triggers_single_detector_HM.TriggerList>`

Summary
-------

Convenience function to generate hole-corrected overlaps on the whole file The first three parameters are prefered in order of appearance

Signature
---------

.. code-block:: python

   def gen_overlaps(self, wfs_whitened_fd = None, calphas = None, random_calpha = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wfs_whitened_fd``
     - -
     - None
     - Array of n_wf x 3 x len(rfftfreq(fftsize)) with FFT of waveform (can be a vector if n_wf = 1)
   * - ``calphas``
     - -
     - None
     - Array of n_wf x calphas with coefficients of basis for template (can be a vector if n_wf = 1)
   * - ``random_calpha``
     - -
     - True
     - Flag to query the generator for a random template

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

   Convenience function to generate hole-corrected overlaps on the
   whole file
   The first three parameters are prefered in order of appearance
   :param wfs_whitened_fd:
       Array of n_wf x 3 x len(rfftfreq(fftsize)) with FFT of waveform
       (can be a vector if n_wf = 1)
   :param calphas:
       Array of n_wf x calphas with coefficients of basis for template
       (can be a vector if n_wf = 1)
   :param random_calpha: Flag to query the generator for a random template
   :return:
