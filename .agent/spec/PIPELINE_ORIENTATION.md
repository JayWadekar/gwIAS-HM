# GWIAS-HM Pipeline Orientation

This document is a collaborator-facing orientation guide for the codebase.
For a fuller human-readable narrative walkthrough, see:
- `/Users/tejaswi/Work/gwIAS-HM/PIPELINE_OVERVIEW.md`

## 1) Big-Picture Goal
This repository implements a gravitational-wave (GW) search pipeline aimed at compact binary mergers with higher harmonics (HM), especially mode content beyond the quadrupole-only (22) approximation.

The design objective is to recover HM sensitivity gains without paying the full brute-force computational cost of naive HM template filtering.

The core strategy is:
- compressed template representation in basis coefficients (`calpha`),
- mode-by-mode matched filtering (22, 33, 44),
- robust coincidence/veto/coherent scoring/ranking in real non-Gaussian detector noise.

## 2) Three-Engine Architecture
Think of the pipeline as three connected engines:

1. Single-detector trigger engine
- preprocess and condition strain data,
- matched-filter over template grids,
- generate corrected/interpolated trigger candidates.

2. Multi-detector coincidence/coherence engine
- pair triggers across detectors,
- generate time-slide background,
- apply vetoes and optional coherent marginalization scores.

3. Global ranking engine
- combine subbanks/banks,
- apply prior/sensitivity terms,
- rank candidates against background.

## 3) Stage-by-Stage Workflow

### Stage A: Template banks and priors
Primary files:
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/template_bank_generator_HM.py`
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/template_bank_params_O3a_HM.py`
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/ML_modules.py`
- `/Users/tejaswi/Work/gwIAS-HM/Tutorial_notebooks/1.Template_banks.ipynb`
- `/Users/tejaswi/Work/gwIAS-HM/Tutorial_notebooks/2.Astrophysical_prior.ipynb`

What happens:
- build/load subbanks in reduced coordinates (`calpha`),
- attach mode-ratio samples and optional learned priors,
- keep filtering tractable by structured bank decomposition.

Why:
- this is the computational bottleneck reducer for HM search.

### Stage B: Data conditioning and trigger generation
Primary files:
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/data_operations.py`
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/readligo.py`
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/triggers_single_detector_HM.py`
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/gw_detect_file.py`
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/triggering_on_cluster.py`

What happens:
- PSD estimation, whitening, notch/line handling,
- hole filling/inpainting and mask management,
- transient and outlier tests (multiple veto-style checks),
- template filtering with staged thresholding and sinc refinement,
- save processed single-detector trigger products.

Why:
- instrument noise is non-stationary and non-Gaussian, so robust conditioning is mandatory before ranking statistics are meaningful.

### Stage C: Coincidence, vetoes, coherent scores
Primary files:
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/coincidence_HM.py`
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/coherent_score_hm_search.py`
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/coherent_score_mz_fast.py`

What happens:
- pair triggers by timing and template identity logic,
- build background via timeslides,
- apply candidate-level veto and optimization passes,
- optionally compute coherent marginalized score terms,
- serialize candidate sets and metadata.

Why:
- convert large single-detector trigger clouds into physically consistent multi-detector candidates with controlled false-alarm behavior.

### Stage D: Ranking and significance
Primary file:
- `/Users/tejaswi/Work/gwIAS-HM/Pipeline/ranking_HM.py`

What happens:
- aggregate candidates across runs/subbanks,
- apply coherent and incoherent terms, template/sensitivity corrections,
- maximize/group to avoid overcounting,
- produce ranked candidate lists and significance-like outputs.

Why:
- this is where pipeline outputs become scientifically actionable event candidates.

## 4) Scientific Lineage Context
The code mixes:
- current HM-search machinery and modern coherent marginalization methods,
- legacy quadrupole-only pipeline components/results inherited from earlier search generations.

The references index contains draft mapping confidence and code relevance notes:
- `/Users/tejaswi/Work/gwIAS-HM/references/REFERENCES.md`

## 5) What a New Contributor Should Read First
Recommended order:
1. `/Users/tejaswi/Work/gwIAS-HM/README.md`
2. `/Users/tejaswi/Work/gwIAS-HM/.agent/spec/PIPELINE_SPEC.md`
3. `/Users/tejaswi/Work/gwIAS-HM/Pipeline/triggers_single_detector_HM.py`
4. `/Users/tejaswi/Work/gwIAS-HM/Pipeline/coincidence_HM.py`
5. `/Users/tejaswi/Work/gwIAS-HM/Pipeline/ranking_HM.py`
6. template/ML notebooks and modules as needed.

## 6) Practical Caveats
- The code includes run/version compatibility paths and legacy behavior toggles.
- Environment and path assumptions exist in utilities and cluster wrappers.
- Some features are explicitly hardcoded (e.g., current dominant mode set and detector assumptions), as also noted in the README TODO list.
