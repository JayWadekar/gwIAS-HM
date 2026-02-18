triggering_on_cluster
=====================

Back to :doc:`API tree index <../index>`

Purpose
-------

Cluster submission and trigger-run orchestration helpers.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`check_file <../functions/triggering_on_cluster.check_file>`
     - No docstring summary available.
   * - :doc:`check_files_server <../functions/triggering_on_cluster.check_files_server>`
     - Warning: doesn't work on Hyperion due to lack to bash commands!
   * - :doc:`clean_file <../functions/triggering_on_cluster.clean_file>`
     - No docstring summary available.
   * - :doc:`create_trig_dir_name <../functions/triggering_on_cluster.create_trig_dir_name>`
     - Old function to create an output directory name for a subbank in a run. The name carries information about when the analysis was launched.
   * - :doc:`filelist <../functions/triggering_on_cluster.filelist>`
     - No docstring summary available.
   * - :doc:`finish_checks <../functions/triggering_on_cluster.finish_checks>`
     - Warning: doesn't work on Hyperion due to lack of bash commands!
   * - :doc:`get_problematic <../functions/triggering_on_cluster.get_problematic>`
     - Checks what files in the directory are excessively large
   * - :doc:`get_strain_filelist <../functions/triggering_on_cluster.get_strain_filelist>`
     - No docstring summary available.
   * - :doc:`inspect_completion <../functions/triggering_on_cluster.inspect_completion>`
     - Inspects the current status of files in the output directory. Useful for checking which files failed, ran partially, etc.
   * - :doc:`submit_files_helios <../functions/triggering_on_cluster.submit_files_helios>`
     - No docstring summary available.
   * - :doc:`submit_files_hyperion <../functions/triggering_on_cluster.submit_files_hyperion>`
     - Hyperion doesn't have as much memory, so suggest using exclusive
   * - :doc:`submit_files_typhon <../functions/triggering_on_cluster.submit_files_typhon>`
     - No docstring summary available.
   * - :doc:`submit_files_wexac <../functions/triggering_on_cluster.submit_files_wexac>`
     - No docstring summary available.
   * - :doc:`submit_multibanks <../functions/triggering_on_cluster.submit_multibanks>`
     - Submits multibanks to the cluster

No public classes were found.

.. toctree::
   :hidden:
   :maxdepth: 1

   ../functions/triggering_on_cluster.check_file
   ../functions/triggering_on_cluster.check_files_server
   ../functions/triggering_on_cluster.clean_file
   ../functions/triggering_on_cluster.create_trig_dir_name
   ../functions/triggering_on_cluster.filelist
   ../functions/triggering_on_cluster.finish_checks
   ../functions/triggering_on_cluster.get_problematic
   ../functions/triggering_on_cluster.get_strain_filelist
   ../functions/triggering_on_cluster.inspect_completion
   ../functions/triggering_on_cluster.submit_files_helios
   ../functions/triggering_on_cluster.submit_files_hyperion
   ../functions/triggering_on_cluster.submit_files_typhon
   ../functions/triggering_on_cluster.submit_files_wexac
   ../functions/triggering_on_cluster.submit_multibanks
