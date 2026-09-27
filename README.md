# COBOL Multimodal Agents

This repository contains the experimental artifacts for a multimodal-agent study on the automated assessment of waste reports. The agents analyze one or more report images together with report metadata and use multimodal large language models to classify the reported waste, assess report-quality properties, and detect potential duplicate reports.

The repository includes the agent implementations, ten train/test splits, controlled noisy-label variants, manually curated ground truth, and the CSV outputs produced during the experiments.


## Repository contents

```text
COBOL_multimodal-agents-main/
├── agents/
│   ├── embed.py                 # Builds the multimodal Chroma retrieval database
│   ├── rq1/                     # Classification-verification agents
│   ├── rq2/                     # Independent-classification agents
│   └── rq3/                     # Duplicate-detection agents
├── cases/
│   ├── clean_cases/             # 10 clean train/test splits
│   ├── noisy size/              # Test splits with perturbed Size labels
│   ├── noisy type/              # Test splits with perturbed ContainedWaste labels
│   └── noisy size type/         # Test splits with both labels perturbed
├── results/
│   ├── rq1/                     # RQ1 clean/noisy outputs and execution times
│   ├── rq2/                     # RQ2 outputs and execution times
│   └── rq3/                     # Duplicate-detection outputs
├── GroundTruth - paper.xlsx     # Reference annotations used for evaluation
└── README.md
```

The archive currently contains 15 Python scripts, 50 case CSV files, and more than 400 precomputed result CSV files.

# Experimental tasks

### RQ1 — Classification verification

The agents in `agents/rq1/` receive the report images together with the user-provided `ContainedWaste`, `Size`, and `Description` fields. They are prompted to verify the submitted classification and independently produce:

- a waste-type classification;
- a waste-size classification;
- a textual rationale;
- `multiple_subjects`;
- `accurate_description`;
- `human_in_frame`;
- `clear_subject`.

RQ1 is evaluated both on the original clean test reports and on controlled noisy variants in which the waste size, waste type, or both have been perturbed.

### RQ2 — Independent classification

The agents in `agents/rq2/` classify the report from the visual evidence without using the submitted waste classification as guidance in the prompt. Their predictions are then stored alongside the original report fields so that they can be compared with the reference classification.

### RQ3 — Duplicate detection

The agents in `agents/rq3/` first retrieve visually related reports and then ask the multimodal model whether the query report is a duplicate of one of the retrieved candidates.

The duplicate-detection output contains:

- `report_id`;
- `duplicate_check`;
- `duplicate_rank`;
- `confidence`;
- `notes`;
- `error`.

The default retrieval depth is `top_k = 3`.

## Agent configurations

For RQ1 and RQ2, six configurations are implemented. The final character identifies the multimodal backbone: `l` for LLaVA and `q` for Qwen2.5-VL.

| Configuration | Backbone | Context strategy |
|---|---|---|
| `L0l` | LLaVA | LLM-only, zero-shot |
| `LFl` | LLaVA | LLM-only, few-shot |
| `R0l` | LLaVA | Retrieval-augmented, zero-shot |
| `L0q` | Qwen2.5-VL | LLM-only, zero-shot |
| `LFq` | Qwen2.5-VL | LLM-only, few-shot |
| `R0q` | Qwen2.5-VL | Retrieval-augmented, zero-shot |

Some result files retain legacy agent letters from earlier experiment runs. For example, RQ1 results use `C = L0l`, `F = L0q`, `H = R0l`, and `L = R0q`; the few-shot outputs are stored directly as `LFl` and `LFq`. RQ2 similarly contains legacy filenames such as `reports_agent_A.csv`, `reports_agent_B.csv`, etc.

## Classification space

The prompts constrain `ContainedWaste` to the following vocabulary:

```text
aluminum/metal
waste not identifiable
construction materials
glass
plastic
textiles
wood
bulky waste
electronic appareil
tyres
paper
chemicals and drugs
organic
other
```

The size vocabulary is:

```text
small
medium
big
```

For waste-type comparison, the scripts emit `CORRECT`, `INCOMPLETE`, or `WRONG`. Size comparison is emitted as `CORRECT` or `WRONG`.

## Data

`cases/clean_cases/` contains ten train/test splits named `split_01` through `split_10`. Each test split contains approximately 30 reports; the remaining reports are used for training/retrieval.

The noisy test sets preserve the report content while perturbing selected submitted labels:

- `cases/noisy size/`: altered `Size`;
- `cases/noisy type/`: altered `ContainedWaste`;
- `cases/noisy size type/`: altered `Size` and `ContainedWaste`.

The CSV files contain report metadata and one or more image URLs in the `Picture` field. Images are not bundled in the repository: the scripts download them when an experiment is executed. Consequently, rerunning the experiments requires network access to the referenced image resources.

## Retrieval pipeline

Retrieval-based agents use a multimodal Chroma database built by `agents/embed.py`.

The embedding pipeline uses OpenCLIP `ViT-L-14` with OpenAI weights. For every training example, the script:

1. downloads the report image(s);
2. encodes the textual report representation with CLIP;
3. encodes the image with CLIP;
4. concatenates the text and image embeddings;
5. stores the resulting vector and report metadata in a persistent Chroma collection named `waste_combined`.

At inference time, retrieval-based agents encode the query image, retrieve the top-k closest examples using cosine distance, and provide the retrieved examples to the multimodal model as additional context.

## Requirements

The repository does not currently include a pinned `requirements.txt` or environment file. The imports used by the scripts require at least the following Python packages:

```bash
pip install \
  numpy \
  pandas \
  requests \
  pillow \
  torch \
  open_clip_torch \
  chromadb \
  langchain-community \
  langchain-core \
  langchain-ollama \
  ollama
```

The experiments use a local [Ollama](https://ollama.com/) server with the following models referenced in the code:

```bash
ollama pull llava:13b
ollama pull qwen2.5vl:7b
```

Start Ollama before running the agents:

```bash
ollama serve
```

A CUDA-capable GPU is not mandatory for all code paths, but it is strongly recommended for CLIP embedding and multimodal-model inference.

## Environment variables

The scripts recognize several environment variables:

| Variable | Purpose | Default in the code |
|---|---|---|
| `OLLAMA_HOST` | Ollama server address | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Model override in scripts that expose it | `llava:13b` or `qwen2.5vl:7b`, depending on the script |
| `FORCE_DEVICE` | Device used by CLIP | `cuda` if available, otherwise `cpu` |
| `HF_HOME` | Hugging Face cache | `~/.cache/huggingface` |
| `TORCH_HOME` | PyTorch cache | `~/.cache/torch` |
| `TOP_K` | Number of retrieved duplicate candidates in RQ3 | `3` |

Note that several Qwen RQ1/RQ2 scripts currently specify `qwen2.5vl:7b` directly inside `ollama.generate()` rather than reading `OLLAMA_MODEL`.

## Reproducing an experiment

The intended workflow for one split is:

1. choose a split, e.g. `split_01`;
2. build the retrieval database from `cases/clean_cases/split_01_train.csv`;
3. run the desired agent on `cases/clean_cases/split_01_test.csv`, or on the corresponding noisy test file for the RQ1 robustness experiment;
4. repeat the procedure for the remaining splits;
5. use the files under `results/` or newly generated CSVs for metric computation and analysis.

Conceptually:

```bash
# 1. Build the retrieval database for the selected training split
python agents/embed.py

# 2. Run one agent, for example the LLaVA RQ1 zero-shot configuration
python agents/rq1/agent_L0l.py

# 3. Run duplicate detection
python agents/rq3/agent_duplicate_l.py
```

### Important path note

The ZIP preserves paths used in the original experiment environment. They are not fully aligned with the packaged repository layout.

In particular:

- the packaged datasets are under `cases/`, while several scripts still reference `reports/cv10_splits`;
- `agents/embed.py` and the RQ-specific scripts do not all point to the same default `chroma_db_combined` location;
- input split names and output filenames are hard-coded in each script's `__main__` block.

Before rerunning the experiments, update `REPORTS_DIR`, `CHROMA_DIR`, the selected `split_XX` path, and the output location so that they point to the desired files in this repository. For retrieval-based evaluation, the Chroma database must be rebuilt from the training partition corresponding to the test split being evaluated.

The files already stored under `results/` are the precomputed experimental outputs and can be inspected without rerunning the models.

## Output files

RQ1 and RQ2 report CSVs append model-generated fields such as:

```text
containedWaste_generated
size_generated
ContainedWaste_match
Size_match
Notes
multiple_subjects
accurate_description
human_in_frame
clear_subject
```

Timing files contain the processed row index and elapsed classification time in seconds. RQ3 stores duplicate decisions separately in `duplicate_check_results_*.csv`.

## Reproducibility notes

- Model outputs can vary across Ollama/model versions, hardware, and decoding implementations.
- The repository does not pin Python package versions or Ollama model digests.
- Report images are fetched from external URLs and therefore depend on continued availability of those resources.
- Retrieval runs depend on the Chroma database generated for the corresponding training split.
- The repository includes precomputed outputs so that the reported experiment artifacts remain inspectable even when exact reruns are not possible.


