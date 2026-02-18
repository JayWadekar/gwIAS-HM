template_bank_generator_HM.TemplateBank.transform_pars
======================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Transforms [m1, m2, s1z, s2z, lambda1, lambda2] to commonly used parameters [mchirp, eta, chieff, chia, tilde{lambda}, delta tilde{lambda}]

Signature
---------

.. code-block:: python

   def transform_pars(pars)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``pars``
     - -
     - -
     - return value of gen_and_save_temp_structure (can be vector for n_sample = 1)

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - n_sample x 6 array with mchirp, eta, chieff, chia, tilde{lambda} and delta tilde{lambda} (can be vector for n_wf = 1)

Docstring
---------

.. code-block:: text

   Transforms [m1, m2, s1z, s2z, lambda1, lambda2] to commonly used
   parameters [mchirp, eta, chieff, chia, tilde{lambda},
   delta tilde{lambda}]
   :param pars:
       return value of gen_and_save_temp_structure
       (can be vector for n_sample = 1)
   :return: n_sample x 6 array with mchirp, eta, chieff, chia,
            tilde{lambda} and delta tilde{lambda}
            (can be vector for n_wf = 1)
