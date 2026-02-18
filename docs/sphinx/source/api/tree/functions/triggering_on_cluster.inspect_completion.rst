triggering_on_cluster.inspect_completion
========================================

Back to :doc:`Module page <../modules/triggering_on_cluster>`

Summary
-------

Inspects the current status of files in the output directory. Useful for checking which files failed, ran partially, etc.

Signature
---------

.. code-block:: python

   def inspect_completion(fnames, output_dir)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fnames``
     - -
     - -
     - List of filenames for strain data (e.g., '../H-H1_GWOSC_O3a_4KHZ_R1-1238355968-4096.hdf5')
   * - ``output_dir``
     - -
     - -
     - Directory where the output files were generated

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - List with entries 0:unprocessed_files, 1:untriggered_files, 2:partial_files 3:failed_files, 4:indeterminate_files, 5:completed_files unprocessed_files: List of files that haven't been preprocessed untriggered_files: List of files that have been preprocessed but not triggered partial_files: Dictionary of files that have been partially triggered (key: filename, value: number of bankchunks done) failed_files: List of files that have failed indeterminate_files: List of files that have an unknown status completed_files: List of files that have been completely triggered

Docstring
---------

.. code-block:: text

   Inspects the current status of files in the output directory. 
   Useful for checking which files failed, ran partially, etc.
   
   :param fnames: List of filenames for strain data
                   (e.g., '../H-H1_GWOSC_O3a_4KHZ_R1-1238355968-4096.hdf5')
   :param output_dir: Directory where the output files were generated
   :return: List with entries
       0:unprocessed_files, 1:untriggered_files, 2:partial_files
       3:failed_files, 4:indeterminate_files, 5:completed_files
       
       unprocessed_files: List of files that haven't been preprocessed
       untriggered_files: List of files that have been preprocessed 
                           but not triggered
       partial_files: Dictionary of files that have been partially triggered
                       (key: filename, value: number of bankchunks done)
       failed_files: List of files that have failed
       indeterminate_files: List of files that have an unknown status
       completed_files: List of files that have been completely triggered
