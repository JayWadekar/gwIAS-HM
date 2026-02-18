# GWIAS-HM Canonical Pipeline Spec

This is the canonical, agent-optimized specification for pipeline aims and architecture.

## Metadata
```yaml
spec_name: GWIAS-HM Canonical Pipeline Spec
spec_version: 1.0.0
last_updated: 2026-02-18
status: active
source_of_truth: true
related_docs:
  orientation: /Users/tejaswi/Work/gwIAS-HM/.agent/spec/PIPELINE_ORIENTATION.md
  todo: /Users/tejaswi/Work/gwIAS-HM/.agent/spec/PIPELINE_TODO.md
  references_index: /Users/tejaswi/Work/gwIAS-HM/references/REFERENCES.md
```

## Mission
```yaml
mission:
  primary_goal: "Search for compact binary GW signals with higher harmonics while controlling computational cost and false alarms."
  strategy:
    - "Template compression into basis coefficients (calpha)."
    - "Mode-by-mode filtering and HM-aware scoring."
    - "Robust multi-detector coincidence, vetoing, coherent scoring, and ranking."
```

## In-Scope Capabilities
```yaml
capabilities:
  - id: template_bank_hm
    description: "Build/load HM-aware template banks and subbanks."
    primary_files:
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/template_bank_generator_HM.py
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/template_bank_params_O3a_HM.py

  - id: single_detector_triggering
    description: "Preprocess data and generate corrected/interpolated single-detector triggers."
    primary_files:
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/data_operations.py
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/triggers_single_detector_HM.py
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/gw_detect_file.py

  - id: cluster_triggering_ops
    description: "Submit, monitor, and recover large triggering campaigns on cluster environments."
    primary_files:
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/triggering_on_cluster.py

  - id: coincidence_and_veto
    description: "Collect detector coincidences, build timeslide background, and apply veto/optimization logic."
    primary_files:
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/coincidence_HM.py

  - id: coherent_scoring
    description: "Compute coherent marginalized score terms for candidate evaluation."
    primary_files:
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/coherent_score_hm_search.py
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/coherent_score_mz_fast.py

  - id: ranking_and_significance
    description: "Aggregate and rank candidate sets across subbanks/runs with priors and sensitivity corrections."
    primary_files:
      - /Users/tejaswi/Work/gwIAS-HM/Pipeline/ranking_HM.py
```

## Pipeline Dataflow
```yaml
dataflow:
  - stage: A_template_bank
    outputs:
      - "Subbank metadata and basis representations"
      - "Mode-ratio samples and optional prior models"

  - stage: B_preprocess_and_trigger
    inputs:
      - "strain files"
      - "template metadata"
    outputs:
      - "single-detector trigger files"
      - "preprocessing artifacts"

  - stage: C_coincidence_and_coherence
    inputs:
      - "per-detector trigger files"
    outputs:
      - "candidate arrays, veto metadata, optional timeseries/coherent terms"

  - stage: D_ranking
    inputs:
      - "candidate products across subbanks/runs"
    outputs:
      - "ranked background/zero-lag/event/injection candidate views"
```

## Scientific Context Tags
```yaml
context_tags:
  - "HM mode-by-mode filtering"
  - "Relative-binning-inspired local refinement"
  - "Coherent marginalization"
  - "Non-Gaussian/non-stationary noise robustness"
  - "Legacy quadrupole lineage with HM-forward direction"
```

## Current Structural Constraints
```yaml
constraints:
  - "Some detector/mode assumptions remain partially hardcoded."
  - "Environment/path assumptions exist in utility and cluster code."
  - "Backward-compatibility branches for old run formats are present."
```

## Update Protocol
```yaml
update_protocol:
  canonical_rule: "This file is the canonical aim/architecture spec."
  todo_sync_rule: "Completed TODO items must be reflected in this spec."
  versioning:
    scheme: "MAJOR.MINOR.PATCH"
    major: "Incompatible conceptual restructuring"
    minor: "New capabilities/aims/workflow sections"
    patch: "Clarifications and non-semantic edits"
  required_on_change:
    - "Bump spec_version"
    - "Update last_updated"
    - "Append changelog entry"
```

## Changelog
- `1.0.0` (2026-02-18): Initial canonical spec created from code walkthrough + references mapping.
