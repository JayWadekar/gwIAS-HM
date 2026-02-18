Getting Started
===============

Here is a systematic narrative of what this pipeline is doing and why.

1. Big Picture Goal
-------------------

This codebase is a gravitational-wave search pipeline optimized for compact binary black holes with higher harmonics (HM), not only quadrupole (22) waveforms.

The practical goal is to recover HM sensitivity gains while keeping the search computationally tractable in real detector noise.

2. Core Architecture
--------------------

The pipeline is organized into three coupled engines:

1. Single-detector triggering
2. Multi-detector coincidence and coherence
3. Global ranking and significance estimation

3. Pipeline Workflow
--------------------

Stage A: Template banks and priors

- Build/load HM subbanks in reduced coordinates (``calpha``)
- Attach mode-ratio samples and optional learned priors
- Reduce brute-force HM filtering cost through structured bank decomposition

Primary modules:

- ``Pipeline/template_bank_generator_HM.py``
- ``Pipeline/template_bank_params_O3a_HM.py``
- ``Pipeline/ML_modules.py``

Stage B: Data conditioning and single-detector triggers

- Estimate PSDs, whiten strain, and handle lines/holes/glitches
- Run matched filtering over templates
- Refine/interpolate trigger candidates and persist trigger products

Primary modules:

- ``Pipeline/data_operations.py``
- ``Pipeline/readligo.py``
- ``Pipeline/triggers_single_detector_HM.py``
- ``Pipeline/gw_detect_file.py``
- ``Pipeline/triggering_on_cluster.py``

Stage C: Coincidence and coherent scoring

- Pair triggers across detectors using timing/template consistency
- Build background with time slides
- Apply vetoes and optional coherent marginalization scores

Primary modules:

- ``Pipeline/coincidence_HM.py``
- ``Pipeline/coherent_score_hm_search.py``
- ``Pipeline/coherent_score_mz_fast.py``

Stage D: Ranking and candidate significance

- Aggregate candidates across banks/subbanks/runs
- Combine incoherent and coherent terms with prior/sensitivity corrections
- Produce ranked outputs for scientific follow-up

Primary module:

- ``Pipeline/ranking_HM.py``

4. Why This Structure Exists
----------------------------

Detector noise is non-stationary and non-Gaussian. The pipeline therefore separates:

- robust conditioning and trigger generation,
- physically consistent multi-detector consistency tests,
- ranking/statistical calibration.

This decomposition keeps sensitivity high while controlling false alarms and runtime cost.

5. What to Read Next
--------------------

1. ``README.md``
2. ``.agent/spec/PIPELINE_SPEC.md``
3. ``Pipeline/triggers_single_detector_HM.py``
4. ``Pipeline/coincidence_HM.py``
5. ``Pipeline/ranking_HM.py``
