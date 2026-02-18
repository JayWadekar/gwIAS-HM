ML_modules
==========

Back to :doc:`API tree index <../index>`

Purpose
-------

Optional ML prior/posterior helpers used by ranking/scoring stages.

Module summary
--------------

No module-level description available.

.. list-table:: Top-level functions
   :header-rows: 1

   * - Function
     - Summary
   * - :doc:`build_mlp <../functions/ML_modules.build_mlp>`
     - Create an multi-layer perceptron (MLP) from the configuration.
   * - :doc:`get_flow <../functions/ML_modules.get_flow>`
     - Instantiate a simple Masked Autoregressive normalizing flow.

.. list-table:: Classes
   :header-rows: 1

   * - Class
     - Summary
   * - :doc:`NeuralPosteriorEstimator <../classes/ML_modules.NeuralPosteriorEstimator>`
     - Simple neural posterior estimator class using a normalizing flow as the posterior density estimator.
   * - :doc:`Template_Prior_NF <../classes/ML_modules.Template_Prior_NF>`
     - param:

.. toctree::
   :hidden:
   :maxdepth: 1

   ../classes/ML_modules.NeuralPosteriorEstimator
   ../classes/ML_modules.Template_Prior_NF
   ../functions/ML_modules.build_mlp
   ../functions/ML_modules.get_flow
