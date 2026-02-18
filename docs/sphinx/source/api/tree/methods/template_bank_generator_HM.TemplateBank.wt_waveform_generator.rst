template_bank_generator_HM.TemplateBank.wt_waveform_generator
=============================================================

Back to :doc:`Class page <../classes/template_bank_generator_HM.TemplateBank>`

Summary
-------

Generator that returns waveforms in bank and corresponding c_alphas

Signature
---------

.. code-block:: python

   def wt_waveform_generator(self, delta_calpha = 0.7, fudge = params.TEMPLATE_SAFETY, remove_nonphysical = True, force_zero = True, ncores = 1, coreidx = 0, orthogonalize = True, return_cov = False, random_order = False)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``delta_calpha``
     - -
     - 0.7
     - Grid point distance in terms of basis coefficients
   * - ``fudge``
     - -
     - params.TEMPLATE_SAFETY
     - Safety factor to inflate the range of used values of each parameter
   * - ``remove_nonphysical``
     - -
     - True
     - Flag indicating whether we want to keep only the gridpoints that are close to a physical waveform
   * - ``force_zero``
     - -
     - True
     - Flag indicating whether we want to center the grid to get as much overlap as possible with the central region
   * - ``ncores``
     - -
     - 1
     - -
   * - ``coreidx``
     - -
     - 0
     - -
   * - ``orthogonalize``
     - -
     - True
     - Orthogonalizes the different mode wfs
   * - ``return_cov``
     - -
     - False
     - Returns the covariance matrix between mode wfs
   * - ``random_order``
     - -
     - False
     - Flag indicating whether we want the waveforms in a random order

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - Generator that returns whitened frequency domain templates (lives on rfftfreq(fftsize, dt)) in units of 1/Hz, and coefficients of basis functions

Docstring
---------

.. code-block:: text

   Generator that returns waveforms in bank and corresponding c_alphas
   :param delta_calpha: Grid point distance in terms of basis coefficients
   :param fudge:
       Safety factor to inflate the range of used values of each parameter
   :param remove_nonphysical:
       Flag indicating whether we want to keep only the gridpoints that
       are close to a physical waveform
   :param force_zero:
       Flag indicating whether we want to center the grid to get as much
       overlap as possible with the central region
   :param ncores:
   :param coreidx:
   :param orthogonalize: Orthogonalizes the different mode wfs
   :param return_cov: Returns the covariance matrix between mode wfs
   :param random_order: Flag indicating whether we want the waveforms in a
                        random order
   :return: Generator that returns whitened frequency domain templates
            (lives on rfftfreq(fftsize, dt)) in units of 1/Hz, and
            coefficients of basis functions
