ranking_HM.maximize_over_groups
===============================

Back to :doc:`Module page <../modules/ranking_HM>`

Summary
-------

-

Signature
---------

.. code-block:: python

   def maximize_over_groups(events_list, incoherent_score_func = utils.incoherent_score, coherent_score_func = utils.coherent_score, n_templates_by_major_bank = None, bank_details_by_subbank = None, template_prior_applied = False, precomputed_details = False, global_maximization_format = 'new')

.. list-table:: Input variables
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - ``events_list``
     - -
     - -
     - List of tuples with (pclists, (chirp_mass_id, subbank id)) if incoherent (prior_terms, pclists, (chirp_mass_id, subbank id)) if coherent
   * - ``incoherent_score_func``
     - -
     - utils.incoherent_score
     - Function that accepts two processedclists for a trigger and returns an incoherent score (the idea was that we can use rank score if we want)
   * - ``coherent_score_func``
     - -
     - utils.coherent_score
     - Function that accepts the coherent terms for a trigger and returns the coherent score (just the sum by default)
   * - ``n_templates_by_major_bank``
     - -
     - None
     - Dictionary giving the number of templates in each chirp mass bank
   * - ``bank_details_by_subbank``
     - -
     - None
     - Dictionary giving the calpha_dimensionality, and the grid spacing of each subbank
   * - ``template_prior_applied``
     - -
     - False
     - Flag indicating whether the template prior was applied
   * - ``precomputed_details``
     - -
     - False
     - Flag to indicate that we precomputed the details using bg_fg_by_subbank, we added an entry to each event info with t_H, t_L, ... (more if more detectors), incoherent score used for ranking, rho^2, and coherent score used for ranking
   * - ``global_maximization_format``
     - -
     - 'new'
     - 'new' for the new format, 'old' for the old format If 'new', we ensure that the maximization over banks/subbanks does not depend on the ordering 'old' was the default for all published catalogs before 08-29-2024 'new' is the recommended input for all future catalogs

Output variables
----------------

.. list-table::
   :header-rows: 1

   * - Return annotation
     - Docstring type
     - Description
   * - ``None``
     - -
     - List in the same format, maximized over pair of 0.1 s groups

Docstring
---------

.. code-block:: text

   :param events_list:
       List of tuples with (pclists, (chirp_mass_id, subbank id)) if incoherent
       (prior_terms, pclists, (chirp_mass_id, subbank id)) if coherent
   :param incoherent_score_func:
       Function that accepts two processedclists for a trigger and returns an
       incoherent score (the idea was that we can use rank score if we want)
   :param coherent_score_func:
       Function that accepts the coherent terms for a trigger and returns
       the coherent score (just the sum by default)
   :param n_templates_by_major_bank:
       Dictionary giving the number of templates in each chirp mass bank
   :param bank_details_by_subbank:
       Dictionary giving the calpha_dimensionality, and the grid spacing of
       each subbank
   :param template_prior_applied:
       Flag indicating whether the template prior was applied
   :param precomputed_details:
       Flag to indicate that we precomputed the details using bg_fg_by_subbank,
       we added an entry to each event info with t_H, t_L, ... (more if more
       detectors), incoherent score used for ranking, rho^2, and coherent score
       used for ranking
   :param global_maximization_format:
       'new' for the new format, 'old' for the old format
       If 'new', we ensure that the maximization over banks/subbanks does not
       depend on the ordering
       'old' was the default for all published catalogs before 08-29-2024
       'new' is the recommended input for all future catalogs
   :return: List in the same format, maximized over pair of 0.1 s groups
