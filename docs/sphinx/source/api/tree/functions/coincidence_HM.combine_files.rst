coincidence_HM.combine_files
============================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Combines files that were read in using collect_files

Signature
---------

.. code-block:: python

   def combine_files(cands_after_veto, cands_before_veto, timeseries)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``cands_after_veto``
     - -
     - -
     - (possibly empty) List of (possibly empty) n_files read in
   * - ``cands_before_veto``
     - -
     - -
     - (possibly empty) list of (possibly empty) n_files read in
   * - ``timeseries``
     - -
     - -
     - (possibly empty) list of (possibly empty) n_files read in

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

   Combines files that were read in using collect_files
   :param cands_after_veto:
       (possibly empty) List of (possibly empty) n_files read in
   :param cands_before_veto:
       (possibly empty) list of (possibly empty) n_files read in
   :param timeseries:
       (possibly empty) list of (possibly empty) n_files read in
   :return:
