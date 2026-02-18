coincidence_HM.fit_step_veto
============================

Back to :doc:`Module page <../modules/coincidence_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def fit_step_veto(trigger, det_ind, f_power = 1)

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``trigger``
     - -
     - -
     - A (trigger, bank_id) pair.
   * - ``det_ind``
     - -
     - -
     - 0 for Hanford, 1 for Livingston.
   * - ``f_power``
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
     - 1. veto_score - higher is worse. 2. best_delta_ts - the relative shift between the GW and the best fit step functions 3. The best scores as function of position withing 16 seconds of the event time 4. Empirical score distribution of the max (fit_step chi2) over position, for 16 seconds near event time

Docstring
---------

.. code-block:: text

   :param trigger: A (trigger, bank_id) pair.
   :param det_ind: 0 for Hanford, 1 for Livingston.
   :return: 1. veto_score - higher is worse.
            2. best_delta_ts - the relative shift between the GW and the best fit step functions
            3. The best scores as function of position withing 16 seconds of the event time
            4. Empirical score distribution of the max (fit_step chi2) over position, for 16 seconds near event time
