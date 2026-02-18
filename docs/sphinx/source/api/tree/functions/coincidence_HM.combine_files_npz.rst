coincidence_HM.combine_files_npz
================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

Combines files that were read in using collect_files_npz

Signature
---------

.. code-block:: python

   def combine_files_npz(candidate_info, apply_vetoes = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``candidate_info``
     - -
     - -
     - (possibly empty) List of (possibly empty) n_files read in
   * - ``apply_vetoes``
     - -
     - False
     - Flag to enforece vetoes when saving candidates

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Combined candidate info, in the form of a list with the elements being candidates, veto_metadata, coherent score (+timeseries) if asked for, with vetoes applied if needed

Docstring
---------

.. code-block:: text

   Combines files that were read in using collect_files_npz
   :param candidate_info:
       (possibly empty) List of (possibly empty) n_files read in
   :param apply_vetoes: Flag to enforece vetoes when saving candidates
   :return:
       Combined candidate info, in the form of a list with the elements being
       candidates, veto_metadata, coherent score (+timeseries) if asked for,
       with vetoes applied if needed
