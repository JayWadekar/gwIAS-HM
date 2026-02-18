coincidence_HM.collect_files_npz
================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Returns list of length nfiles with each element being a tuple with events, mask_vetoed, veto_metadata, coherent_scores (+timeseries if collect_timeseries) an element is None if the byte-size < 10

Signature
---------

.. code-block:: python

   def collect_files_npz(filelist, ncores = 1, collect_timeseries = True)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``filelist``
     - -
     - -
     - -
   * - ``ncores``
     - -
     - 1
     - -
   * - ``collect_timeseries``
     - -
     - True
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

   Returns list of length nfiles with each element being a tuple with
   events, mask_vetoed, veto_metadata, coherent_scores
   (+timeseries if collect_timeseries)
   an element is None if the byte-size < 10
