# The Safety Tax of Post-Training Quantization

Code and results for the paper **"The Safety Tax of Post-Training Quantization: A Cross-Family Behavioral Audit of Small Instruction-Tuned Language Models."**

We compare 12 instruction-tuned checkpoints from 7 families under FP16/BF16, INT8 (LLM.int8) and NF4 (bitsandbytes), using 100 AdvBench-derived harmful prompts x 5 template attacks and 130 benign prompts, greedy decoding (200 new tokens), an LLM judge, and paired statistics (bootstrap, McNemar, Holm).

| Family | Checkpoints | Folder |
|---|---|---|
| Qwen2.5 | 1.5B, 3B | `part1_.../qwen` |
| Llama-3.2 | 1B, 3B | `part1_.../llama` |
| Phi | 3.5-mini, 3-mini-4k | `part1_.../phi` |
| Gemma | 3-1B, 2-2B, 3-4B | `part1_.../gemma` |
| Ministral | 3-3B (official BF16 release) | `part1_.../mistral` |
| SmolLM2 | 1.7B | `part2_.../smollm` |
| Granite | 3.1-2B | `part2_.../granite` |

## Repository layout

```text
README.md  LICENSE  requirements.txt
data/prompts/                       100 harmful base prompts, 130 benign prompts
results/<family>/                   final results per family (tables, stats, audit summaries, label-only row files)
common/cross_family_analysis/       cross-family statistics, tables and figures
common/                             build_results_release.py, verify_release.py
part1_qwen_llama_phi_gemma_mistral/ code for Qwen, Llama, Phi, Gemma, Mistral   (Person 1)
part2_granite_smollm/               code for Granite, SmolLM2                        (Person 2)
```

Inside each family folder the scripts are numbered in the order they were run (`00` generation, then the later fix/audit stages). **The highest number is the final stage.** Scripts are Kaggle notebooks converted to Python (`# %%` cells) and can be run with `python <script>.py` or cell by cell.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
```

A CUDA GPU is only needed for generation. Everything else runs on CPU.

API keys are read from environment variables (never stored in the repo): `HF_TOKEN`, `GEMINI_API_KEY` (or `GCP_SERVICE_ACCOUNT_JSON`, `GCP_PROJECT`, `GCP_LOCATION`), `GROQ_API_KEY` (Granite judge).

The per-family scripts use the Kaggle paths from the study (`/kaggle/input/...`, `/kaggle/working/...`); edit the path constants at the top of a script to run elsewhere.

## Reproducing the cross-family results

The final results are in `results/`, so no GPU or API key is needed:

```bash
cd common/cross_family_analysis
python load.py            # checks 100 bases x 5 attacks x 3 precisions per checkpoint
python analysis.py        # cluster bootstrap + permutation test, Holm across checkpoints
python primary_family.py
python identity_check.py
python mkforest.py        # figures
python tables.py          # LaTeX tables
```

## Re-running a family from scratch

Run the family's scripts in numeric order (for example `llama/00_...` then `01_...` then `02_...`). Generation needs a GPU and Hugging Face access to the model; judging needs the judge API key.

## What is in `results/`

Per family: summary, statistics, safety-tax-delta, benign and human-corrected tables, audit agreement and confusion matrices, claim-check / validation / provenance reports, the run manifest, and label-only row-level files (`judged_harmful_<family>_labels.csv`, `judged_benign_<family>_labels.csv`). Model-generated text (responses, audit sheets) is not included, because it contains harmful content. `results/VERIFICATION_REPORT.md` lists the automated checks run on each package, and `python common/build_results_release.py --include-all` rebuilds the folder from the final packages (`PACKAGES_ROOT` points to the unzipped packages).

## Contributing

Each contributor edits only their own folder (`part1_...` or `part2_...`). Pull before you push.

## License

MIT, see `LICENSE`. AdvBench prompts are from Zou et al. (2023); please check the upstream license before redistributing them.
