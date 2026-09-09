# Evidence-Grounded Agentic Multimodal Verification of Bangla News Photocards

This repository contains the code, dataset subset, and evaluation
artefacts for the paper:

> Anam, M., Nahid, K. H., & Rahaman, A. S. M. M. (2026).
> Evidence-Grounded Agentic Multimodal Verification of Bangla News
> Photocards Using Vision--Language Models and Web Retrieval.
> *Applied Intelligence* (under review).

The framework combines a three-agent CrewAI pipeline (Extraction,
Retrieval, Judge) with a misattribution-aware three-stage retrieval
strategy that explicitly separates source-attribution impersonation
from outright fabrication. It is instantiated on two vision--language
backends (Qwen2.5-VL-7B via Ollama and Gemini-3.5-Flash via API) and
evaluated on 1,000 balanced photocards from the BanglaFakeCard
benchmark.

---

## Repository contents

| Directory | Contents |
|---|---|
| `code/` | All source code: baseline scripts, agentic pipeline, custom tools, McNemar test, evaluation notebook |
| `dataset/` | The 1,000-photocard subset used in the paper (500 real, 500 fake) with ground-truth annotations |
| `reference_logos/` | Canonical logo library organised one folder per outlet |
| `results/` | Per-instance CSV outputs for all four experimental conditions |
| `docs/` | Diagrams and supplementary documentation |

---

## Installation

Tested on Windows 11 with Python 3.11 and a single NVIDIA GPU with
at least 12 GB of VRAM.

```bash
git clone https://github.com/[YOUR_USERNAME]/BanglaFakeCard-Agentic-Verification.git
cd BanglaFakeCard-Agentic-Verification
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The Qwen2.5-VL-7B backend is served locally through Ollama. Install
Ollama from https://ollama.com and pull the model:

```bash
ollama pull qwen2.5vl:7b
```

The Gemini-3.5-Flash backend is accessed through the Google Gen AI
API. Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

You will need:
- `GOOGLE_API_KEY` — for Gemini-3.5-Flash
- `SERPER_API_KEY_1`, `SERPER_API_KEY_2`, ... — one or more Google
  Serper API keys (multi-key rotation is supported)

---

## Reproducing the paper results

Each of the four experimental conditions writes results row-by-row to
a CSV and can resume from an interrupted run.

**Qwen baseline:**
```bash
python code/qwen_baseline_llm_final.py
```

**Gemini baseline:**
```bash
python code/gemini_baseline_llm_final.py
```

**Qwen agentic:**
```bash
PIPELINE_MODE=open_source python code/agentic_workflow_final.py
```

**Gemini agentic:**
```bash
PIPELINE_MODE=api python code/agentic_workflow_final.py
```

**Statistical significance test:**
```bash
python code/mcnemar_test_final.py
```

**Full evaluation report** (all metrics, tables, and figures):
```bash
jupyter notebook code/model_evaluation_report_final.ipynb
```

Expected runtime for the full 1,000-photocard evaluation is
approximately 3–5 hours per condition on a single consumer GPU
(Qwen conditions bottleneck on local inference; Gemini conditions
bottleneck on API latency).

---

## Dataset

The `dataset/` directory contains the 1,000-photocard subset used in
this study, drawn from the BanglaFakeCard benchmark. Please see
`dataset/README.md` for full attribution, licence terms, and
composition details.

**Original dataset citation:**

> Ali, E., Emon, A., Mahmud, S. A., Faruk, O., Fahim, M. F. A., &
> Patwary, M. J. A. (2026). BanglaFakeCard: A Multimodal Dataset for
> Bangla News Photocard Verification.

---

## Citation

If you use this code, the dataset, or the results in your research,
please cite our paper:

```bibtex
@article{anam2026banglafakecard,
  title   = {Evidence-Grounded Agentic Multimodal Verification of Bangla
             News Photocards Using Vision-Language Models and Web Retrieval},
  author  = {Anam, Mahfuz and Nahid, Kamrul Hasan and Rahaman, Abu Sayed Md. Mostafizur},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the original BanglaFakeCard dataset paper (see above
and `dataset/README.md`).

---

## Licences

This repository uses two separate licences:

- **All code** in the `code/` directory is released under the MIT
  Licence. See `LICENSE`.
- **The dataset** in the `dataset/` directory is redistributed under
  the original BanglaFakeCard licence. See `DATASET_LICENSE`.
- **All outlet logos and news photographs** embedded in the
  photocards remain the intellectual property of their respective
  owners and are included solely for academic reproducibility of the
  results reported in the paper.

---

## Contact

For questions or issues, please open a GitHub issue or contact the
first author at [mahfuzanam181@gmail.com].
