utils.env_init_lines
====================

Back to :doc:`Module page <../modules/utils>`

Summary
-------

Creates text snippet to be added before submitting jobs to clusters

Signature
---------

.. code-block:: python

   def env_init_lines(env_command = None, module_name = None, env_name = None, python_name = None, cluster = 'helios')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``env_command``
     - -
     - None
     - Override the combination of module loading and sourcing the conda environment
   * - ``module_name``
     - -
     - None
     - Override module to be initialized
   * - ``env_name``
     - -
     - None
     - Override user-dependent conda environment name
   * - ``python_name``
     - -
     - None
     - Override user- and cluster-dependent python command name
   * - ``cluster``
     - -
     - 'helios'
     - Cluster being used, add options below for new clusters

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Text snippet that is to be added in the commands to the cluster (no ending space)

Docstring
---------

.. code-block:: text

   Creates text snippet to be added before submitting jobs to clusters
   :param env_command:
       Override the combination of module loading and sourcing the conda
       environment
   :param module_name: Override module to be initialized
   :param env_name: Override user-dependent conda environment name
   :param python_name: Override user- and cluster-dependent python command name
   :param cluster: Cluster being used, add options below for new clusters
   :return:
       Text snippet that is to be added in the commands to the cluster
       (no ending space)
