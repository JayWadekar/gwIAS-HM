template_bank_generator_HM.get_prior_interp_func
================================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

create a function that interpolate the prior for N-dimensional c-alpha array

Signature
---------

.. code-block:: python

   def get_prior_interp_func(mb_key, subbank, ndim = 2, prior_data = None, prior_data_file_path = None)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``mb_key``
     - -
     - -
     - str, multibank key, see template_bank_params ('BBH_0', 'BBH_1', etc).
   * - ``subbank``
     - -
     - -
     - int, sub-bank index within multibank. Put 0 if the multibank contain a single sub-bank.
   * - ``ndim``
     - -
     - 2
     - int, dimension of input (default prior_data files are made with 5, so ndim<=5)
   * - ``prior_data``
     - -
     - None
     - prior_data dictionary, with structure: (properties dictionary)->(multibank dictionary)->(sub-bank-list).
   * - ``prior_data_file_path``
     - -
     - None
     - path str. if prior_data not passed, will load from here.

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - 1. a function that gets c_alpha array and interpolate it's prior, according to the scheme, 2. Number of interpolation dimensions to be used

Docstring
---------

.. code-block:: text

   create a function that interpolate the prior for N-dimensional c-alpha array
   :param mb_key: str, multibank key, see template_bank_params ('BBH_0', 'BBH_1', etc).
   :param subbank: int, sub-bank index within multibank. 
                   Put 0 if the multibank contain a single sub-bank.
   :param ndim: int, dimension of input 
               (default prior_data files are made with 5, so ndim<=5)
   :param prior_data: prior_data dictionary, with structure:
    (properties dictionary)->(multibank dictionary)->(sub-bank-list).
   :param prior_data_file_path: path str. if prior_data not passed, will load from here.
   :return: 
       1. a function that gets c_alpha array and interpolate it's prior,
          according to the scheme, 
       2. Number of interpolation dimensions to be used
