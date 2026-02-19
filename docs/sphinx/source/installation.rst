Installation and Environment
============================

This page summarizes how to set up an environment for running GWIAS-HM.

Required packages
-----------------

Core dependencies from the repository README:

- ``numpy``
- ``scipy``
- ``matplotlib``
- ``scikit-learn``
- ``pandas``
- ``gsl``
- ``lal``
- ``lalsimulation``
- ``astropy``
- ``numba``
- ``multiprocess``
- `cogwheel <https://github.com/jroulet/cogwheel>`_

Optional but recommended
------------------------

- ``torch``
- `lightning <https://lightning.ai/docs/pytorch/stable/starter/installation.html>`_
- `nflows <https://pypi.org/project/nflows/>`_
- ``corner``
- ``jupyterlab``

Suggested setup workflow
------------------------

1. Create and activate your environment manager of choice (for example Conda).
2. Install the required scientific stack listed above.
3. Install optional ML/notebook packages if you plan to run prior-learning workflows.
4. Verify that key imports succeed from the repository root.

Quick import check
------------------

From the repository root, run:

.. code-block:: bash

   python - <<'PY'
   import numpy, scipy, pandas, matplotlib, astropy, numba, multiprocess
   print("Core Python dependencies import successfully.")
   PY

Notes
-----

- Some workflows rely on LIGO-related libraries and data-access tooling.
- Cluster workflows and large-scale runs may need additional site-specific configuration.
- For documentation building requirements specifically, see ``docs/sphinx/requirements.txt``.
