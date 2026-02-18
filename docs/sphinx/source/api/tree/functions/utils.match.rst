utils.match
===========

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Computes match, or cosine, between waveforms

Signature
---------

.. code-block:: python

   def match(wfs_cos_1, wfs_cos_2, allow_shift = True, allow_phase = True, return_cov = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``wfs_cos_1``
     - -
     - -
     - n_wf x len(wf) array with whitened waveforms (can be vector if n_wf = 1)
   * - ``wfs_cos_2``
     - -
     - -
     - n_wf x len(wf) array with whitened waveforms (can be vector if n_wf = 1)
   * - ``allow_shift``
     - -
     - True
     - Flag to allow shifts when computing the match
   * - ``allow_phase``
     - -
     - True
     - Flag to allow phases when computing the match
   * - ``return_cov``
     - -
     - False
     - Flag to return timeseries of complex match

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - If return cov, timeseries of complex match between the waveforms, else the match or cosine (defined as \\\|complex match\\\|)

Docstring
---------

.. code-block:: text

   Computes match, or cosine, between waveforms
   :param wfs_cos_1:
       n_wf x len(wf) array with whitened waveforms (can be vector if n_wf = 1)
   :param wfs_cos_2:
       n_wf x len(wf) array with whitened waveforms (can be vector if n_wf = 1)
   :param allow_shift: Flag to allow shifts when computing the match
   :param allow_phase: Flag to allow phases when computing the match
   :param return_cov: Flag to return timeseries of complex match
   :return:
       If return cov, timeseries of complex match between the waveforms, else
       the match or cosine (defined as |complex match|)
