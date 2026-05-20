# BirdCLEF 2026 — Autonomous Research Agent

An autonomous AI research agent that iteratively designs, trains, evaluates, and improves deep learning models for bird species audio classification (BirdCLEF 2026, Track B).

The project trains **two backbones**: EfficientNet (repo root) and YamNet (`Yamnet runs/`).

## Run everything with one command

```bash
conda activate keras_env     # the environment you trained in
./run.sh
```

`run.sh` does the full pipeline end-to-end:

1. Installs the pinned dependencies (`requirements.txt`) and verifies TF 2.19.0 / Keras 3.10.0
2. Starts Ollama and pulls the LLM (`gemma4:e4b`)
3. Links the shared `data/` into `Yamnet runs/` (the YamNet agent reads data relative to its own folder)
4. Runs the **EfficientNet** agent, then the **YamNet** agent — full research runs

> ⏱️ This is the full run (`DEBUG=False`): expect **several hours per backbone** on CPU.
> Output streams to the console and is saved per run under `logs/`. If one agent
> errors, the script logs it and still runs the other.
>
> 📦 The YamNet backbone is bundled offline at `Yamnet runs/models/yamnet_savedmodel/`,
> so no download is needed at run time.

## How it works

Each iteration the agent runs this loop:

1. **Prompt** — Sends the LLM (Ollama `gemma4:e4b`) a description of the backbone and all past experiment results
2. **Generate** — LLM writes a Keras model head (Dense layers only; backbone is fixed)
3. **Validate** — Basic checks before execution: uses `backbone_model`, no class definitions, no forbidden calls
4. **Train** — Phase 1 on `train_audio`; Phase 2 on soundscape windows if val_auc plateaus
5. **Analyse** — LLM reflects on the result and explains what to try next
6. **Log** — Result appended to `experiments.jsonl` (never overwritten)
7. **Save** — Best model by val_auc saved to `models/best_model.keras`
8. **Repeat** — History fed back into the next prompt

## Setup

> ⚠️ **Versions are not optional.** The Kaggle BirdCLEF 2026 image runs
> **TensorFlow 2.19.0 / Keras 3.10.0**. A newer Keras writes model-config keys
> that Kaggle's older Keras cannot load, so a model trained on the wrong
> versions *silently fails on Kaggle*. `requirements.txt` pins these exactly,
> and `agent.py` refuses to run on a mismatch. Always install via
> `pip install -r requirements.txt` — do not `pip install tensorflow` loose.

```bash
# 1. Create conda environment (Python 3.12 — matches Kaggle)
conda create -n keras_env python=3.12 -y
conda activate keras_env

# 2. Install pinned dependencies (TF 2.19.0 / Keras 3.10.0)
pip install -r requirements.txt

# 3. Verify versions match Kaggle
python -c "import tensorflow as tf; print(tf.__version__, tf.keras.__version__)"
#   expected:  2.19.0 3.10.0

# 4. Install and start Ollama  (https://ollama.com)
ollama pull gemma4:e4b
ollama serve   # runs in background
```

### Optional: Apple Silicon GPU acceleration

On an Apple Silicon Mac (M1/M2/M3) you can train on the GPU via Metal —
much faster, and it does **not** affect Kaggle `.keras` compatibility
(that depends on the Keras version, not the device).

```bash
# Uncomment the tensorflow-metal line in requirements.txt, then:
pip install tensorflow-metal==1.2.0

# Sanity-check it BEFORE a long run: set DEBUG=True in agent.py and run once.
# Loss should decrease normally. If pip rejects the version, or you see
# NaNs / garbage loss / unsupported-op errors, fall back:
pip uninstall -y tensorflow-metal
pip install tensorflow-metal          # let pip pick the build for TF 2.19
#   then re-pin requirements.txt to whatever version it installed
```

Intel Macs: skip this entirely (Metal is Apple-Silicon only; CPU path works).

## Data layout

Place the BirdCLEF 2026 competition files under `data/`:

```
data/
  train.csv
  taxonomy.csv
  train_audio/          # individual recordings (.ogg)
  train_soundscapes/    # soundscape recordings (.ogg)
  train_soundscapes_labels.csv
```

## Running the agent

```bash
conda activate keras_env
python agent.py
```

Key config knobs at the top of `agent.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `N_ITERATIONS` | 5 | Number of LLM-propose-train cycles |
| `N_EPOCHS` | 5 | Max epochs per iteration |
| `BACKBONE` | `"efficientnet"` | Pretrained feature extractor |
| `FINE_TUNE` | `True` | Unfreeze backbone weights |
| `DEBUG` | `False` | Use 64 samples for fast pipeline checks |

## Viewing results

```python
from experiment_log import print_summary
print_summary()
```

## Kaggle submission

1. After training, upload `models/best_model.keras` to Kaggle as a dataset named `birdclef-model`
2. Open `submission.ipynb` in Kaggle, add your dataset via **Add Data → Your Datasets**
3. Run all cells — outputs `submission.csv`

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Main agent loop |
| `prompt_builder.py` | Builds LLM prompts with experiment history |
| `experiment_log.py` | Append-only JSONL experiment logger |
| `submission.ipynb` | Kaggle inference notebook |
| `experiments.jsonl` | All experiment results (auto-generated) |
| `models/best_model.keras` | Best model weights (auto-generated) |
