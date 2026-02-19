Tutorial Notebooks
==================

The repository includes notebooks to help users understand and reproduce major stages of the pipeline.

Notebook sequence
-----------------

1. ``Tutorial_notebooks/1.Template_banks.ipynb``
   Build higher-harmonic template banks from scratch.
2. ``Tutorial_notebooks/2.Astrophysical_prior.ipynb``
   Generate astrophysical priors for templates and HM mode-amplitude ratios.
3. ``Tutorial_notebooks/3.Trig_Coin_test_with_injection.ipynb``
   Test triggering and coincidence modules on injected examples.
4. ``Tutorial_notebooks/4.Trig_Coin_on_cluster.ipynb``
   Run triggering/coincidence workflows on bulk observing-run data in a cluster setting.
5. ``Tutorial_notebooks/5.Ranking_candidates.ipynb``
   Combine candidates across banks/subbanks and compute ranking/FAR-style outputs.
6. ``Tutorial_notebooks/6.p_astro+detection_tables.ipynb``
   Compute ``p_astro`` and detection-table style outputs.

External notebook source
------------------------

The README notes that the most up-to-date notebook copies are also available in this shared folder:

- `GWIAS notebooks on Google Drive <https://drive.google.com/drive/folders/15avuKxY40aX9Ru_6xkacM1ZdaGL7OgQm?usp=sharing>`_

Suggested usage
---------------

1. Start with ``1.Template_banks.ipynb`` and ``3.Trig_Coin_test_with_injection.ipynb`` to build intuition.
2. Move to ``4.Trig_Coin_on_cluster.ipynb`` only after local understanding of trigger/coincidence flow.
3. Finish with ranking/astrophysical interpretation notebooks.
