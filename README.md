# Target Prioritization Engine

A data integration platform that consolidates genomic, chemical, clinical, and proteomic evidence into a unified target intelligence table — supporting strategic R&D prioritization, competitive benchmarking, and drug development decision-making.

## Overview

This pipeline integrates four major biomedical data sources to build `targets_ml` — a feature-rich table describing drug targets, their associated drugs, mechanisms of action, and clinical trial activity:

| Source | Role |
|---|---|
| **Open Targets** | Maps Ensembl gene IDs to UniProt SwissProt accessions |
| **ChEMBL** | Provides human single-protein targets, drug mechanisms, and drug metadata |
| **ClinicalTrials.gov** | Links drugs to clinical trials by name matching |
| **UniProt** (optional) | Adds protein annotation to the enriched output table |

## Pipeline Flow


## Strategic Use Cases

- **Target prioritization**: Identify targets with strong drug/mechanism evidence and clinical validation
- **Competitive landscape mapping**: See which targets have approved drugs vs. early pipeline activity
- **Due diligence support**: Quickly assess clinical trial activity and development stage for a given target
- **Portfolio gap analysis**: Surface targets with mechanisms but limited clinical progression

## Installation

```bash
pip install -r requirements.txt
python build_targets_ml.py \
    --opentargets-target-dir open_targets/target \
    --chembl-db chembl/chembl_37/chembl_37_sqlite/chembl_37.db \
    --trials clinical_trials/clinicaltrials_all.jsonl \
    --uniprot-tsv uniprot/uniprot_human.tsv \
    --out-dir results
