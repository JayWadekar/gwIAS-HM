# GWIAS-HM Pipeline Overview

## 1. Big Picture Goal
This codebase is a gravitational-wave search pipeline that is optimized for compact binary black holes with higher harmonics (HM), not just quadrupole (22) waveforms.

Core objective:
- keep the sensitivity gains of higher modes,
- avoid the brute-force computational explosion that usually comes with HM template banks.

That is why the pipeline is built around:
- a compressed template representation (basis coefficients `calpha`),
- mode-by-mode matched filtering (22, 33, 44 separately),
- a staged scoring/ranking system that controls false alarms in non-Gaussian detector noise.

Primary implementation roots:
- [`Pipeline/`](Pipeline/)
- [`Tutorial_notebooks/`](Tutorial_notebooks/)

## 2. Statistical Detection Principle
High-level statistical strategy:
- detection is framed in Neyman-Pearson terms (likelihood-ratio style statistic),
- strict Neyman-Pearson optimality is for simple-vs-simple hypotheses,
- in the composite case (unknown source parameters), the intended extension is to marginalize over model parameters, i.e. use an evidence-like statistic.

Current implementation caveat:
- extrinsic-parameter marginalization is treated explicitly in coherent-score modules,
- intrinsic-parameter marginalization is not yet fully rigorous end-to-end; in current practice it is handled in an approximate/semi-marginalized way consistent with the Appendix discussion in [arXiv:1904.07214](https://arxiv.org/abs/1904.07214).

## 3. Three-Engine Mental Model
Think of the search as three engines:
1. single-detector trigger production,
2. multi-detector coincidence/coherent consistency,
3. global ranking and significance assignment.

## 4. Stage-by-Stage Narrative

### 3.1 Template-bank construction and representation
Implemented mainly in:
- [`Pipeline/template_bank_generator_HM.py`](Pipeline/template_bank_generator_HM.py)

Supporting config/notebooks:
- [`Pipeline/template_bank_params_O3a_HM.py`](Pipeline/template_bank_params_O3a_HM.py)
- [`Tutorial_notebooks/1.Template_banks.ipynb`](Tutorial_notebooks/1.Template_banks.ipynb)

Key behavior:
- bank is represented in reduced coordinates (`calpha`) rather than raw physical parameters during filtering,
- subbanks partition parameter space to make filtering tractable,
- HM amplitude-ratio samples are attached to banks for downstream coherent/ranking stages.

Why:
- this is the computational trick that makes HM search feasible at scale.

### 3.2 Data ingestion and preprocessing
Implemented mainly in:
- [`Pipeline/data_operations.py`](Pipeline/data_operations.py)
- [`Pipeline/readligo.py`](Pipeline/readligo.py)

Steps include:
- PSD estimation,
- whitening/filter conditioning,
- line finding/notching,
- hole filling/inpainting,
- multiple transient/glitch tests (outliers, sine-Gaussian, excess power).

Why:
- real interferometer data is non-stationary and non-Gaussian, so robust preprocessing is essential before matched filtering.

### 3.3 Single-detector filtering and trigger generation
Main class:
- `TriggerList` in [`Pipeline/triggers_single_detector_HM.py`](Pipeline/triggers_single_detector_HM.py)

Entrypoints:
- [`Pipeline/gw_detect_file.py`](Pipeline/gw_detect_file.py) (CLI)
- [`Pipeline/triggering_on_cluster.py`](Pipeline/triggering_on_cluster.py) (cluster launcher)

Core behavior:
- enumerate templates over `calpha` grid,
- compute mode-wise overlaps,
- apply hole and PSD-drift corrections,
- use base threshold to find candidate regions,
- sinc-interpolate to finer timing,
- save processed triggers with calibrated fields.

Why staged thresholds:
- cheap coarse reject first,
- expensive interpolation only where needed.

HM-specific note:
- pipeline can use marginalized HM score terms to reject unphysical mode-ratio behavior near threshold.

### 3.4 Coincidence building and veto optimization
Main module:
- [`Pipeline/coincidence_HM.py`](Pipeline/coincidence_HM.py)

What it does:
- load per-detector trigger files by epoch,
- build coincident candidates using timing buckets and template IDs,
- generate background via time slides,
- optimize trigger representations locally,
- apply veto logic and attach metadata explaining veto outcomes.

Output format:
- newer `.npz` outputs can include candidates, veto masks, metadata keys, optional timeseries, and optional coherent scores.

Why:
- this stage turns many single-detector triggers into a manageable candidate population with explicit background estimation.

### 3.5 Coherent score computation
Newer HM implementation:
- [`Pipeline/coherent_score_hm_search.py`](Pipeline/coherent_score_hm_search.py)

Older fast/legacy structure:
- [`Pipeline/coherent_score_mz_fast.py`](Pipeline/coherent_score_mz_fast.py)

Concept:
- use local detector timeseries around each candidate,
- marginalize over extrinsic parameters (sky, polarization, etc.) with QMC/integration machinery,
- incorporate HM amplitude-ratio sampling.

Why:
- coherent marginalization better separates astrophysical consistency from accidental coincidences.

### 3.6 Ranking and significance estimation
Main module:
- [`Pipeline/ranking_HM.py`](Pipeline/ranking_HM.py)

It:
- collects subbank outputs,
- splits into background, zero-lag, known-event overlap, and injections,
- combines incoherent and coherent terms,
- applies astrophysical/learned prior information where configured,
- applies veto policies,
- maximizes over subbanks/banks to avoid double counting,
- produces ranked candidate lists and IFAR-like significance outputs.

Prior-handling implementations used at this stage include:
- [`Pipeline/ML_modules.py`](Pipeline/ML_modules.py)
- [`Tutorial_notebooks/2.Astrophysical_prior.ipynb`](Tutorial_notebooks/2.Astrophysical_prior.ipynb)

Why:
- final scientific products are ranked events with defensible background-calibrated significance, and this is where the composite-hypothesis evidence-style logic is operationalized in practice.

## 5. New-Collaborator Reading Order
1. [`README.md`](README.md)
2. [`Pipeline/triggers_single_detector_HM.py`](Pipeline/triggers_single_detector_HM.py)
3. [`Pipeline/coincidence_HM.py`](Pipeline/coincidence_HM.py)
4. [`Pipeline/ranking_HM.py`](Pipeline/ranking_HM.py)
5. [`Pipeline/template_bank_generator_HM.py`](Pipeline/template_bank_generator_HM.py)
6. notebooks:
   - [`Tutorial_notebooks/1.Template_banks.ipynb`](Tutorial_notebooks/1.Template_banks.ipynb)
   - [`Tutorial_notebooks/3.Trig_Coin_test_with_injection.ipynb`](Tutorial_notebooks/3.Trig_Coin_test_with_injection.ipynb)
   - [`Tutorial_notebooks/5.Ranking_candidates.ipynb`](Tutorial_notebooks/5.Ranking_candidates.ipynb)

## 6. One-Paragraph Summary
This repository is an HM-capable GW search pipeline that uses compressed template coordinates and mode-by-mode filtering to make HM searches computationally practical, then uses coincidence, vetoing, coherent marginalization, and bank-aware ranking to convert raw triggers into statistically ranked candidate events.
