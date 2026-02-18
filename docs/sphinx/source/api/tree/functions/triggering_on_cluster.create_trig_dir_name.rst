triggering_on_cluster.create_trig_dir_name
==========================================

Back to :doc:`Module page <../modules/triggering_on_cluster>`

Summary
-------

Old function to create an output directory name for a subbank in a run. The name carries information about when the analysis was launched.

Signature
---------

.. code-block:: python

   def create_trig_dir_name(template_conf, base_path = None, run_str = 'O2')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``template_conf``
     - -
     - -
     - Path to the metadata.json file for the subbank
   * - ``base_path``
     - -
     - None
     - Path to the root directory within which the output directories for all subbanks will be located. If None, we will use the default for the run specified in utils.py
   * - ``run_str``
     - -
     - 'O2'
     - String identifying the run (e.g., "O2", "O3a")

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Path to the output directory

Docstring
---------

.. code-block:: text

   Old function to create an output directory name for a subbank in a run.
   The name carries information about when the analysis was launched.
   :param template_conf: Path to the metadata.json file for the subbank
   :param base_path:
       Path to the root directory within which the output directories for all
       subbanks will be located. If None, we will use the default for the run
       specified in utils.py
   :param run_str: String identifying the run (e.g., "O2", "O3a")
   :return: Path to the output directory
