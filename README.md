# BirdCLEF 2026 — Autonomous Research Agent

An autonomous AI research agent that iteratively designs, trains, evaluates, and improves deep learning models for bird species audio classification (BirdCLEF 2026, Track B).

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

```bash
# 1. Create conda environment
conda create -n keras_env python=3.12 -y
conda activate keras_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install and start Ollama  (https://ollama.com)
ollama pull gemma4:e4b
ollama serve   # runs in background
```

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
