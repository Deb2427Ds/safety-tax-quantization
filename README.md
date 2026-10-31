<div align="center">

# The Safety Tax of Post-Training Quantization

### A Cross-Family Behavioral Audit of Small Instruction-Tuned Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Checkpoints](https://img.shields.io/badge/checkpoints-12-8A2BE2.svg)](#evaluated-models)
[![Responses](https://img.shields.io/badge/judged%20responses-22%2C680-success.svg)](#experimental-design)
[![Human audit](https://img.shields.io/badge/human--audited%20rows-4%2C693-orange.svg)](#human-audit-and-validation)

Official code and released results for the paper
**"The Safety Tax of Post-Training Quantization: A Cross-Family Behavioral Audit of Small Instruction-Tuned Language Models."**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Experimental Design](#experimental-design)
- [Evaluated Models](#evaluated-models)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Reproducing the Results](#reproducing-the-results)
- [Released Results](#released-results)
- [Human Audit and Validation](#human-audit-and-validation)
- [Responsible Release](#responsible-release)
- [Limitations](#limitations)
- [Contributing and Code Ownership](#contributing-and-code-ownership)
- [License and Third-Party Material](#license-and-third-party-material)

---

## Overview

Post-training quantization (PTQ) is the default route for deploying small language models (SLMs), yet its effect on **safety alignment** is rarely measured. This repository accompanies a controlled, paired study of the **Safety Tax** of PTQ: the change in harmful compliance, refusal, leakage after refusal, and benign utility caused by lowering weight precision from FP16/BF16 to INT8 or NF4.

Numerical precision is the **only independent variable**. Prompts, greedy decoding, and the judge are held fixed, and every prompt is paired across precision arms.

The study addresses four research questions:

| | Research question |
|---|---|
| **RQ1** | Does INT8 or NF4 quantization change strict harmful compliance, and is the effect consistent across model families? |
| **RQ2** | Through which behavioral channel does any change occur: direct compliance, loss of refusal, or leakage after refusal? |
| **RQ3** | How does the effect depend on the jailbreak strategy and on the model's baseline compliance? |
| **RQ4** | Is there an accompanying cost in benign utility or over-refusal, and are the automated labels reliable under human audit? |

## Key Findings

- **NF4 raises strict harmful compliance in a minority of checkpoints.** Four of twelve checkpoints show a Holm-significant increase; the largest is Gemma-3-1B (+13.8 pp, 18.4% to 32.2%). **INT8 produces no significant change in any checkpoint.**
- **The damage is mostly leakage after refusal.** Where the aggregate effect is significant, no-conflict (direct) compliance stays flat or falls: models still begin by refusing, but a larger share go on to disclose actionable content. A refusal-keyword metric would score these responses as safe.
- **SmolLM2-1.7B reverses the mechanism.** Total compliance is unchanged, but refusal falls by 17.4 pp and direct compliance rises by 10.6 pp.
- **Effects are heterogeneous.** They depend strongly on the jailbreak strategy (for example +28 pp under obfuscation for Qwen2.5-1.5B) and on baseline compliance (which ranges from 2.0% to 64.2%).
- **Benign cost is small.** Over-refusal stays near zero; benign utility falls markedly only for the two Qwen2.5 checkpoints. Judge-human agreement on audited samples is high.

> **Recommendation.** Safety audits of quantized models should report compliance, refusal, and leakage **separately** and across **several attacks**, rather than a single headline rate.

## Experimental Design

| Component | Specification |
|---|---|
| **Checkpoints** | 12 instruction-tuned models from 7 families (about 1-4B parameters) |
| **Precision arms** | FP16/BF16 reference, INT8 (LLM.int8 via `bitsandbytes`), NF4 (4-bit NormalFloat with double quantization, bfloat16 compute) |
| **Quantization** | Data-free; no calibration, fine-tuning, or quantization-aware training |
| **Harmful prompts** | 100 AdvBench-derived base prompts x 5 template attacks = 500 per model-precision cell |
| **Attack strategies** | Direct, DAN, Roleplay/Authority, Hypothetical, Obfuscated |
| **Benign prompts** | 130 prompts for over-refusal and utility |
| **Decoding** | Greedy (`do_sample=False`), 200 new tokens, seed 42 |
| **Scale** | 18,000 harmful + 4,680 benign = **22,680 judged responses** |
| **Judge** | Structured LLM judge: Gemini-2.5 Flash (six families, via Vertex AI) and `mimo-v2.6-flash-free` (Granite) |
| **Statistics** | Cluster bootstrap over base prompts (10,000 resamples), sign-flip permutation test (200,000), McNemar test, Holm correction across the twelve checkpoints |

### Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| **Strict HCR** | Mean of `C` | Actionable harmful assistance, whether or not the response also refuses |
| **No-conflict HCR** | Mean of `C and not R` | Direct jailbreak success with no co-occurring refusal |
| **Refusal rate** | Mean of `R` | Explicit refusal of the harmful request |
| **Leakage (HLR)** | Mean of `L` | Harmful content disclosed after a refusal |
| **Over-refusal (ORR)** | Mean of `O` (benign) | Benign requests refused |
| **Utility (US)** | Mean of `u` in [0, 1] (benign) | Usefulness of benign answers |

The **Safety Tax** of a metric `M` under quantization `Q` in {INT8, NF4} is `ΔM = M_Q − M_FP16`, reported in percentage points.

## Evaluated Models

| Family | Checkpoint | Hugging Face identifier | Code |
|---|---|---|---|
| Qwen2.5 | 1.5B, 3B | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` | `part1_*/qwen` |
| Llama-3.2 | 1B, 3B | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` | `part1_*/llama` |
| Phi | 3.5-mini, 3-mini-4k | `microsoft/Phi-3.5-mini-instruct`, `microsoft/Phi-3-mini-4k-instruct` | `part1_*/phi` |
| Gemma | 3-1B, 2-2B, 3-4B | `google/gemma-3-1b-it`, `google/gemma-2-2b-it`, `google/gemma-3-4b-it` | `part1_*/gemma` |
| Ministral | 3-3B (official BF16 release) | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | `part1_*/mistral` |
| SmolLM2 | 1.7B | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `part2_*/smollm` |
| Granite | 3.1-2B | `ibm-granite/granite-3.1-2b-instruct` | `part2_*/granite` |

> Several of these models are gated on Hugging Face; accept each model's license and provide an `HF_TOKEN` before running generation.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .github/
│   └── CODEOWNERS                           # per-folder ownership
├── data/
│   └── prompts/                             # 100 harmful base prompts, 130 benign prompts (+ SHA256SUMS)
├── common/
│   ├── build_results_release.py             # rebuilds results/ from the final family packages
│   ├── verify_release.py                    # automated integrity checks on each package
│   └── cross_family_analysis/               # cross-family statistics, tables, and figures
│       ├── load.py  analysis.py  primary_family.py
│       └── identity_check.py  mkforest.py  tables.py
├── part1_qwen_llama_phi_gemma_mistral/      # generation, judging, and audit code (Part 1)
│   └── qwen/  llama/  phi/  gemma/  mistral/
├── part2_granite_smollm/                    # generation, judging, and audit code (Part 2)
│   └── granite/  smollm/
└── results/                                 # released, label-only results
    ├── <family>/                            # tables, statistics, audit summaries, reports
    ├── SHA256SUMS
    └── VERIFICATION_REPORT.md
```

Inside each family folder, scripts are **numbered in the order they were run** (`00` generation and judging, then the later fix, post-processing, and audit stages). **The highest-numbered script is the final stage.** Scripts are Kaggle notebooks exported to Python (`# %%` cells) and can be run with `python <script>.py` or cell by cell.

## Getting Started

**Requirements:** Python 3.10 or newer. A CUDA GPU is needed **only for generation**; the analysis runs on CPU.

```bash
git clone <repository-url>
cd <repository-folder>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Package versions are intentionally unpinned in the root `requirements.txt`. For the exact environment used for Granite, see `part2_granite_smollm/granite/requirements.txt`; each family's environment snapshot ships with its release package.

### Credentials

Secrets are read from environment variables and are **never stored in this repository**.

| Variable | Used for |
|---|---|
| `HF_TOKEN` | Downloading gated Hugging Face models |
| `GEMINI_API_KEY` | Gemini-2.5 Flash judge (alternative to the service-account route below) |
| `GCP_SERVICE_ACCOUNT_JSON`, `GCP_PROJECT`, `GCP_LOCATION` | Gemini-2.5 Flash judge via Vertex AI |

```bash
export HF_TOKEN="..."
export GEMINI_API_KEY="..."
```

> **Granite judging.** The final Granite release was judged by `mimo-v2.6-flash-free` (stage `01_granite_v2_pipeline.py`) and reads the stored mimo judgments, so it needs no judge key to re-run. The earlier Gemini-judged pass (`00_granite_historical_gemini_judging.py`) is kept for provenance only, and the earlier Groq-judged results were dropped and are not part of the released statistics.

## Reproducing the Results

### Option A: Reproduce the cross-family analysis (no GPU, no API keys)

The final label-only results are already in `results/`, so the cross-family statistics can be regenerated directly:

```bash
cd common/cross_family_analysis
python load.py              # validates 100 bases x 5 attacks x 3 precisions per checkpoint
python analysis.py          # cluster bootstrap + permutation test, Holm across checkpoints
python primary_family.py    # primary-family (NF4 vs FP16) analysis
python identity_check.py    # identity checks on the analysis outputs
python mkforest.py          # forest-plot figures
python tables.py            # LaTeX tables
```

### Option B: Re-run a family from scratch

Run that family's scripts in numeric order (for example `llama/00_...`, then `01_...`, then `02_...`). Generation needs a GPU and Hugging Face access to the model; judging needs the relevant judge credentials.

The per-family scripts keep the Kaggle paths used in the study (`/kaggle/input/...`, `/kaggle/working/...`). To run elsewhere, edit the path constants at the top of each script, or set the overrides it reads (for example `WORK_DIR`, `OUT_DIR`, `INPUT_ROOT`).

### Option C: Rebuild the released `results/` folder

```bash
export PACKAGES_ROOT=/path/to/unzipped/family/packages
python common/build_results_release.py --include-all
```

This also runs `common/verify_release.py`, which writes `results/VERIFICATION_REPORT.md`.

## Released Results

Each `results/<family>/` folder contains:

- Summary, statistics, safety-tax-delta, and benign-utility tables
- Human-corrected metric summaries and deltas
- Audit agreement tables and confusion matrices
- Claim-check, validation, and provenance reports
- The run manifest
- Label-only row-level files: `judged_harmful_<family>_labels.csv` and `judged_benign_<family>_labels.csv`

`results/SHA256SUMS` and `data/prompts/SHA256SUMS` let you verify file integrity:

```bash
cd results && sha256sum -c SHA256SUMS
```

## Human Audit and Validation

Every family has a stratified human audit that deliberately oversamples the rows most likely to contain judge errors (leakage-after-refusal, compliance/refusal conflicts, provider-blocked outputs, and attack x precision coverage). In total, **4,693 responses** were audited. Human labels replace judge labels on audited rows only.

Each audit used a single annotator, so agreement is reported as **judge-human agreement**, not inter-annotator reliability. Audit provenance for each family is documented in `results/<family>/audit_provenance.md`.

## Responsible Release

This repository releases **labels only**. To avoid redistributing harmful content:

- Raw model responses and human-audit sheets are **not** included.
- Only `data/prompts` and the label-only CSVs under `results/` are tracked; `.gitignore` excludes other data files, figures, and credentials.
- The harmful prompts are AdvBench-derived and are provided solely for reproducing this safety evaluation.

This work is intended to **improve the safety auditing of compressed models**. It does not provide new attack capability.

## Limitations

- Results characterize **data-free** INT8 and NF4 quantization through `bitsandbytes`; they do not necessarily transfer to calibration-based schemes such as GPTQ or AWQ.
- Attacks are fixed **template-based** strategies. We make no claims about robustness to adaptive or optimization-based attackers.
- Decoding is a single greedy configuration, and the study covers models of roughly 1-4B parameters.
- Non-significant results are treated as **inconclusive**, not as evidence of no effect.
- Labels come from an LLM judge validated on audited subsets; each audit used one annotator.

See Section 6 and the appendices of the paper for the full discussion.

## Contributing and Code Ownership

Ownership is defined in [`.github/CODEOWNERS`](.github/CODEOWNERS). Each contributor edits only their own area, so pushes do not collide, and changes to a folder are reviewed by its owner.

| Area | Owner |
|---|---|
| `part1_qwen_llama_phi_gemma_mistral/` | **Deboparna Das** ([@Deb2427Ds](https://github.com/Deb2427Ds)) |
| `results/qwen`, `results/llama`, `results/phi`, `results/gemma`, `results/mistral` | **Deboparna Das** ([@Deb2427Ds](https://github.com/Deb2427Ds)) |
| `part2_granite_smollm/` | **Ishan Dutta** ([@id1509](https://github.com/id1509)) |
| `results/granite`, `results/smollm` | **Ishan Dutta** ([@id1509](https://github.com/id1509)) |
| `common/`, `data/`, `.github/`, top-level `results/` files | Both |

Please pull before you push, and open a pull request for changes to a folder you do not own.

## License and Third-Party Material

This code is released under the [MIT License](LICENSE).

- Harmful prompts are derived from **AdvBench** (Zou et al., 2023). Check the upstream license before redistributing them.
- Model weights are subject to their respective licenses on Hugging Face and are not distributed here.

## Citation

The paper is under anonymous review. A citation entry will be added after the review period.
