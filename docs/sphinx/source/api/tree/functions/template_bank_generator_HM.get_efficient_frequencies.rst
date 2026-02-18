template_bank_generator_HM.get_efficient_frequencies
====================================================

Back to :doc:`Module page <../modules/template_bank_generator_HM>`

Summary
-------

Return a grid of frequencies that allows to safely get unwrapped phase. At low frequencies (fmin < f < fmid) it guarantees that the post-Newtonian phase increases by constant amounts of <= delta_radians for mchirp >= mchirp_min. At high frequencies (fmid < f < fmax) it switches to a regular grid.

Signature
---------

.. code-block:: python

   def get_efficient_frequencies(fmin = 24, fmid = 128, fmax = 512, mchirp_min = 2 ** (-0.2), delta_radians = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``fmin``
     - -
     - 24
     - -
   * - ``fmid``
     - -
     - 128
     - -
   * - ``fmax``
     - -
     - 512
     - -
   * - ``mchirp_min``
     - -
     - 2 \*\* (-0.2)
     - -
   * - ``delta_radians``
     - -
     - 1
     - -

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - -

Docstring
---------

.. code-block:: text

   Return a grid of frequencies that allows to safely
   get unwrapped phase.
   At low frequencies (fmin < f < fmid) it guarantees that
   the post-Newtonian phase increases by constant amounts of
   <= delta_radians for mchirp >= mchirp_min.
   At high frequencies (fmid < f < fmax) it switches to a
   regular grid.
