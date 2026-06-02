<div align="center">
  <img width="800" alt="EVID-Bench" src="assets/main.png" />
  <br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](#-quick-start) [![GitHub Repo](https://img.shields.io/badge/GitHub-EVID--Bench-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/yutao1024/EVID-Bench) [![arXiv](https://img.shields.io/badge/arXiv-coming_soon-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](#-citation) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/Kirito-Lab/EVID-Bench/)

  ✨ **Search-grounded video misinformation detection: retrieve external video evidence, compare across sources, and explain what was falsified.** ✨

  | 🔎 **Fully Automated DeepSearch** | 🧾 **Evidence-Grounded Verification** | 🎬 **9 Manipulation Types** |
</div>




## 📑 Table of Contents



[🔥 News](#-news) | [🌈 Overview](#-overview) | [🤗 Dataset](#-dataset) | [🚀 Quick Start](#-quick-start) | [🧪 Run EVID-Bench](#-run-evid-bench) | [📦 Outputs](#-outputs) | [🛠 Troubleshooting](#-troubleshooting) | [📖 Citation](#-citation)



## 🔥 News

**[2026/06/05]** 📄 Paper will be available on arXiv. Paper link: `PLACEHOLDER_ARXIV_URL`.

**[2026/06/02]** 🤗 EVID-Bench dataset is now available on Hugging Face! Explore the [Dataset](https://huggingface.co/datasets/Kirito-Lab/EVID-Bench/).

**[2026/06/02]** 🚀 EVID-Bench code is released on [GitHub](https://github.com/yutao1024/EVID-Bench).


## 🌈 Overview

<div align="center">
  <img width="800" alt="overview" src="assets/overview.png" />
</div>
<br/>

EVID-Bench studies a setting where a video cannot be verified from pixels alone. A system must search the open web for related videos, compare the input with retrieved evidence, and identify the false or manipulated information.

The benchmark contains **222 manipulated videos** spanning **9 manipulation types** across **3 categories** and **6 topic domains**. The baseline in this repository implements a retrieval-augmented verification pipeline:

```text
Input video
  -> CoT frame analysis
  -> DeepSearch loop: search -> download -> coarse filter -> fine contrastive extraction
  -> sufficiency / reflection
  -> multi-judge scoring against ground truth
```

### Manipulation Taxonomy

- **AI Generation:** identity swap, synthetic insertion, object manipulation.
- **Single-Source Editing:** selective omission, causal inversion, manipulative montage.
- **Multi-Source Editing:** narrative fabrication, magnitude manipulation, contextual fabrication.

## 🤗 Dataset

Download from Hugging Face: [EVID-Bench Dataset](https://huggingface.co/datasets/Kirito-Lab/EVID-Bench/).

Put the manifest and forged videos in **one folder** (name it however you like). The CLI accepts either that folder or the manifest file directly:

```text
data/EVID-Bench/              # example local root
├── Edit.json                 # default manifest name (or any *.json path)
├── NX7QNWEGcNI.mp4           # forged video; filename stem = video ID in the manifest
└── ...
```

Manifest format: a JSON **array** of entries. Minimal fields:

```json
{
  "id": "NX7QNWEGcNI",
  "task": "2.1",
  "topic": "sports",
  "video2": "SOURCE_VIDEO_ID",
  "video3": "",
  "groundtruth": ["First manipulated point.", "Second manipulated point."]
}
```

- **`id`**: forged video ID; must match the `.mp4` basename in the same folder (here `NX7QNWEGcNI.mp4`).
- **`video2` / `video3`** (and other `videoN`): expected source YouTube IDs for retrieval evaluation.
- **`groundtruth`**: forgery points as a string list.

## 🚀 Quick Start

These steps assume a fresh machine and are intentionally explicit.

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)

### Prerequisites

Before installation, install:

- **Python** 3.11+ (Conda recommended)
- **FFmpeg** and `ffprobe`
- **Node.js**, Deno, or Bun (for `yt-dlp` YouTube challenge handling)

macOS:

```bash
brew install ffmpeg node
```

Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg nodejs npm
```

If you do not have root access on a server, install `ffmpeg` and Node.js in your home directory and add their `bin` directories to `PATH`.

### Installation

#### 1. Clone this repository and navigate to the folder

```bash
git clone https://github.com/yutao1024/EVID-Bench.git
cd EVID-Bench
```

#### 2. Create a Conda environment

```bash
conda create -n evid-bench python=3.11 -y
conda activate evid-bench
```

#### 3. Install Python dependencies

```bash
pip install -r requirements.txt
python -m src.cli --help
```

### Configuration

Copy the environment template:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=your_openai_or_openrouter_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
SERPER_KEY=your_serper_key
```

Optional YouTube cookies are helpful when `yt-dlp` sees age gates or bot checks:

```bash
YT_DLP_COOKIES_FILE=/absolute/path/to/cookies.txt
```

## 🧪 Run EVID-Bench

### Minimal One-Video Run

Use this when the manifest and matching `.mp4` files are in the same folder (example: `data/EVID-Bench/`):

```bash
python -m src.cli data/EVID-Bench/Edit.json \
  --only-ids NX7QNWEGcNI \
  --num-workers 1 \
  --candidate-video-height 480 \
  --max-deepsearch-rounds 6 \
  --total-sample-frames 64 \
  --save-run-trace \
  --results-jsonl struct/results.jsonl \
  --trace-jsonl struct/run_trace.jsonl \
  --no-resume \
  --skip-summary \
  --judge-models "gpt-5.5,gemini-3.1-pro-preview,claude-sonnet-4.6" \
  --reasoning-model google/gemini-2.5-pro
```

Change `data/EVID-Bench/` and `NX7QNWEGcNI` to your local folder and sample ID.

### Folder Run

If the folder already contains `Edit.json` (default manifest name) and `<id>.mp4` files:

```bash
python -m src.cli data/EVID-Bench/
```

### Search-Only Retrieval Check

This mode does not download candidates or run visual comparison. It only checks whether search results hit expected source IDs from the manifest.

```bash
python -m src.cli data/EVID-Bench/ --search-only
```

### Common Arguments


| Argument                              | Default                                            | Description                                                                           |
| ------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `input_path`                          | required                                           | Folder containing videos and manifest, or a direct manifest JSON path.                |
| `--manifest`                          | `Edit.json`                                        | Manifest filename when `input_path` is a folder.                                      |
| `--output`                            | `_results`                                         | Output directory under the data folder. `_results` expands to `_results_<timestamp>`. |
| `--only-ids`                          | all                                                | Comma-separated video IDs to run.                                                     |
| `--limit`                             | all                                                | Run only the first `N` manifest entries.                                              |
| `--num-workers` / `--parallel-videos` | `1`                                                | Number of videos to process in parallel.                                              |
| `--candidate-video-height`            | `480`                                              | Frame/video resize height for candidates and cached sampled frames.                   |
| `--total-sample-frames`               | `64`                                               | Input frames for CoT, fine extraction, sufficiency, and reflection.                   |
| `--coarse-sample-frames`              | `16`                                               | Frames per side for coarse relevance filtering.                                       |
| `--candidate-sample-frames`           | same as total                                      | Candidate frames for fine extraction.                                                 |
| `--max-deepsearch-rounds`             | `6`                                                | Maximum DeepSearch search rounds.                                                     |
| `--max-reflect-rounds`                | `3`                                                | Maximum reflection rounds after search failure.                                       |
| `--reasoning-model`                   | `google/gemini-2.5-pro`                            | VLM used for Stage A reasoning and comparison.                                        |
| `--judge-models`                      | `gpt-5.5,gemini-3.1-pro-preview,claude-sonnet-4.6` | Comma-separated Stage C judge panel.                                                  |
| `--judge-model`                       | unset                                              | Legacy single-judge override.                                                         |
| `--save-run-trace`                    | off                                                | Write structured trace rows.                                                          |
| `--results-jsonl`                     | `results.jsonl`                                    | Results JSONL path, relative paths resolved under output dir.                         |
| `--trace-jsonl`                       | `run_trace.jsonl`                                  | Trace JSONL path, relative paths resolved under output dir.                           |
| `--resume` / `--no-resume`            | resume                                             | Reuse or ignore rows already in `results.jsonl`.                                      |
| `--skip-judge`                        | off                                                | Run retrieval only, skip ground-truth judging.                                        |
| `--skip-summary`                      | off                                                | Do not write summary files.                                                           |
| `--download-dir`                      | `downloads`                                        | Candidate download directory, resolved from current working directory.                |


## 📦 Outputs

For a manifest run such as `data/EVID-Bench/Edit.json`, outputs are written next to the manifest:

```text
data/EVID-Bench/
└── _results_YYYYMMDD_HHMMSS/
    ├── results.jsonl
    ├── run_trace.jsonl          # only with --save-run-trace
    ├── summary.json             # skipped by --skip-summary
    ├── summary.md               # skipped by --skip-summary
    └── logs/
        └── run_<timestamp>_<video_id>.log
```

Frame caches are stored under the repository root:

```text
.frame_cache/
├── input/<video_id>_h<height>_n<num_frames>/
└── candidate/<candidate_id>_h<height>_n<num_frames>/
```

These cache and result directories are ignored by git.

## 🗂 Repository Layout

```text
.
├── src/
│   ├── cli.py                 # batch entrypoint
│   ├── agent_pipeline_v2.py   # DeepSearch orchestration
│   ├── agent/
│   │   ├── tools.py           # VLM calls, Serper search, yt-dlp, frame utilities
│   │   ├── prompts.py         # prompts for CoT, filtering, extraction, reflection
│   │   ├── judge.py           # multi-judge scoring
│   │   ├── session_state.py
│   │   └── oracle_eval.py     # manifest and oracle helpers
│   └── utils/
│       ├── config.py          # .env and runtime configuration
│       ├── frame_sampling.py
│       ├── pipeline_log.py
│       ├── token_usage.py
│       └── text_utils.py
├── assets/
│   ├── main.png
│   └── overview.png
├── requirements.txt
├── .env.example
└── README.md
```

## 🛠 Troubleshooting

**`OPENAI_API_KEY` is not set.**  
Create `.env` from `.env.example` and fill `OPENAI_API_KEY`.

**Search returns no candidates.**  
Check `SERPER_KEY` in `.env` (API key from https://serper.dev).

**`ffmpeg` or `ffprobe` not found.**  
Install `ffmpeg` and make sure both commands are on `PATH`.

**YouTube download fails with 403, age checks, or bot checks.**  
Install Node.js / Deno / Bun, update `yt-dlp`, and provide cookies with `YT_DLP_COOKIES_FILE`, `YT_DLP_COOKIES_FILES`, or `YT_DLP_COOKIES_FROM_BROWSER`.

**The run is slow or expensive.**  
Start with `--only-ids`, `--limit 1`, `--candidate-video-height 360`, and `--skip-judge`. Increase coverage after the first end-to-end run works.

**An agent cannot find the output files.**  
Remember that default `--output _results` becomes `_results_<timestamp>`, and relative `--results-jsonl` / `--trace-jsonl` paths are resolved inside that output directory.

## 📖 Citation

Please kindly cite our paper if you find this project or dataset helpful.

```bibtex

```
