coincidence_HM.veto_orthogonal_step
===================================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def veto_orthogonal_step(f_grid, strain_whitened_fd, whitening_filter_fd, best_waveform_fd, double_step = False, use_cosine_steps = True, delta_t_max = 0.015, N_margin = 8192, dt = 1 / 1024.0, index_of_interest = None, f_power = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``f_grid``
     - -
     - -
     - -
   * - ``strain_whitened_fd``
     - -
     - -
     - -
   * - ``whitening_filter_fd``
     - -
     - -
     - -
   * - ``best_waveform_fd``
     - -
     - -
     - -
   * - ``double_step``
     - -
     - False
     - if True, two step functions would be used.
   * - ``use_cosine_steps``
     - -
     - True
     - if True, would fit also the Hilbert transform of a step.
   * - ``delta_t_max``
     - -
     - 0.015
     - -
   * - ``N_margin``
     - -
     - 8192
     - -
   * - ``dt``
     - -
     - 1 / 1024.0
     - dt of whitened strain samples.
   * - ``index_of_interest``
     - -
     - None
     - index around which to search for the highest overlap (Important to give, as can auto tune to an unrelated location!)
   * - ``f_power``
     - -
     - 1
     - will use a waveform that is 1/f\\\*\\\*f_power. default is 1

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

   :param f_grid:
   :param strain_whitened_fd:
   :param whitening_filter_fd:
   :param best_waveform_fd:
   :param double_step: if True, two step functions would be used.
   :param use_cosine_steps: if True, would fit also the Hilbert transform of a step.
   :param delta_t_max:
   :param N_margin:
   :param dt: dt of whitened strain samples.
   :param index_of_interest: index around which to search for the highest overlap (Important to give, as can auto tune to an unrelated location!)
   :param f_power: will use a waveform that is 1/f**f_power. default is 1
   :return:
