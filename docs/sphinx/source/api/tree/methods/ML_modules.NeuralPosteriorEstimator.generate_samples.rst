ML_modules.NeuralPosteriorEstimator.generate_samples
====================================================

Back to :doc:`Class page <../classes/ML_modules.NeuralPosteriorEstimator>`

Summary
-------

Generate samples from of mode amplitude ratios (e.g., R33=A33/A22) as a function of the input calpha.

Signature
---------

.. code-block:: python

   def generate_samples(self, calpha, num_samples = 2048, set_seed = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``calpha``
     - -
     - -
     - np.array The input calpha to generate samples for.
   * - ``num_samples``
     - -
     - 2048
     - int The number of samples to generate.
   * - ``set_seed``
     - -
     - False
     - bool Whether to set the seed for reproducibility.

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - [R33_samples, R44_samples, weight_samples] the weights correspond to the sensitive volume of the samples

Docstring
---------

.. code-block:: text

   Generate samples from of mode amplitude ratios (e.g., R33=A33/A22) 
   as a function of the input calpha.
   :param calpha: np.array
       The input calpha to generate samples for.
   :param num_samples: int
       The number of samples to generate.
   :param set_seed: bool
       Whether to set the seed for reproducibility.
   :return: [R33_samples, R44_samples, weight_samples]
           the weights correspond to the sensitive volume of the samples
