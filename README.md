# BirdCLEF 2026 — Autonomous Research Agent

An autonomous research agent that uses a **locally-hosted LLM** (via Ollama) to
iteratively design, train, and evaluate deep-learning model heads for the
**BirdCLEF 2026** audio classification task: multi-label recognition of **234
species** (birds, amphibians, insects, mammals, reptiles) in 5-second soundscape
windows from the Pantanal wetlands, scored by **macro-averaged ROC-AUC**.

Instead of hand-tuning a single network, the agent runs a closed loop —
*propose → train → evaluate → reflect → repeat* — on top of a **frozen pretrained
backbone**, accumulating an experiment log that conditions each next proposal.

## What we did

- **Two backbones, same agent.** We ran the loop on two frozen feature
  extractors and compared them:
  - **EfficientNetB0** (ImageNet) over **mel-spectrograms** — repo root.
  - **YAMNet** (Google AudioSet) over **raw waveforms** → 1024-d embeddings —
    `Yamnet runs/`. YAMNet is bundled offline so it needs no download at run time.
- **LLM-designed heads.** Each iteration the LLM writes only the classifier head
  (the backbone is fixed); the agent validates the code, trains it, scores it,
  and feeds the result back into the next prompt.
- **Techniques explored across iterations:** focal loss for class imbalance,
  squeeze-and-excitation attention, backbone fine-tuning, varied hidden
  activations, **mixup** (EfficientNet) and **waveform augmentation +
  embedding masking** (YAMNet).
- **Validation that tracks the leaderboard.** A **soundscape-validation regime**
  (validate on held-out soundscapes, the scored distribution) was added to
  reduce the gap between offline scores and the Kaggle leaderboard.
- **Kaggle-compatibility hardening.** The whole stack is pinned to Kaggle's exact
  **TensorFlow 2.19.0 / Keras 3.10.0** so locally-trained `.keras` models load on
  Kaggle without serialization-version errors; `agent.py` refuses to run on a mismatch.
- **Reproducible delivery.** A single `run.sh` sets up the environment and runs
  both backbones end-to-end; `eda.ipynb` and `results.ipynb` document the data
  and the results.

## Results (public leaderboard, macro ROC-AUC)

| Submission | Backbone | Public LB |
|---|---|---|
| Zero-filled baseline | — | 0.512 |
| First head | EfficientNet | 0.632 |
| Soundscape-validation | EfficientNet | 0.711 |
| Augmentation (mixup) | EfficientNet | 0.715 |
| Augmentation | YAMNet | 0.733 |
| **First head** | **YAMNet** | **0.745** |

Best offline validation reached **~0.93** macro-AUC, well above the leaderboard —
the gap is the **domain shift** between clean focal training clips and noisy,
multi-species Pantanal soundscapes (see *Limitations*).

## Quick start — one command

```bash
conda activate keras_env     # the environment you trained in
./run.sh
```

`run.sh` runs the full pipeline:

1. Installs pinned dependencies (`requirements.txt`) and verifies TF 2.19.0 / Keras 3.10.0
2. Starts Ollama and pulls the LLM (`gemma4:e4b`)
3. Links the shared `data/` into `Yamnet runs/` (the YAMNet agent reads data relative to its own folder)
4. Runs the **EfficientNet** agent, then the **YAMNet** agent

> ⏱️ This is the full run (`DEBUG=False`): expect **several hours per backbone** on
> CPU. Output streams to the console and to `logs/`. If one agent errors, the
> script logs it and still runs the other.

## How the agent loop works

Each iteration:

1. **Prompt** — the LLM receives the backbone description and all past results
2. **Generate** — the LLM writes a Keras head (the backbone stays frozen)
3. **Validate** — static checks before execution (uses `backbone_model`, no forbidden calls)
4. **Train & evaluate** — fit the head, measure validation macro ROC-AUC
5. **Reflect** — the LLM analyses the result and proposes what to try next
6. **Log** — appended to `experiments.jsonl` (append-only)
7. **Save** — the best model by validation AUC is written to `models/best_model.keras`

Two training regimes are available via the `SOUNDSCAPE_VAL` toggle: validate on
held-out soundscapes (default for EfficientNet), or clip-validation with a
plateau-gated soundscape phase (default for YAMNet).

## The two backbones

| | EfficientNet (root) | YAMNet (`Yamnet runs/`) |
|---|---|---|
| Pretraining | ImageNet | AudioSet (bioacoustic) |
| Input | mel-spectrogram, 32 kHz, 64 mel bins, 5 s | raw waveform, 16 kHz → 1024-d embedding |
| Iterations × epochs | 8 × 20 | 20 × 40 |
| Augmentation | mixup (α=0.4, p=0.5) | waveform variants (×6) + embedding masking |
| `SOUNDSCAPE_VAL` | `True` | `False` |
| Batch size / LLM | 32 / `gemma4:e4b` | 32 / `gemma4:e4b` |

## Setup (manual)

> ⚠️ **Versions are not optional.** Kaggle's BirdCLEF 2026 image runs
> **TensorFlow 2.19.0 / Keras 3.10.0**. A newer Keras writes model-config keys
> Kaggle's older Keras cannot load, so a model trained on the wrong versions
> *silently fails on Kaggle*. Always install via `requirements.txt`.

```bash
# 1. Conda environment (Python 3.12 — matches Kaggle)
conda create -n keras_env python=3.12 -y
conda activate keras_env

# 2. Pinned dependencies (TF 2.19.0 / Keras 3.10.0)
pip install -r requirements.txt

# 3. Verify versions match Kaggle  → expected: 2.19.0 3.10.0
python -c "import tensorflow as tf, keras; print(tf.__version__, keras.__version__)"

# 4. Local LLM
ollama pull gemma4:e4b
ollama serve   # background
```

### Optional: Apple Silicon GPU

On an Apple-Silicon Mac, Metal can speed up training and does **not** affect
Kaggle `.keras` compatibility (that depends on the Keras version, not the device).
GPU is **off by default**; uncomment `tensorflow-metal` in `requirements.txt`,
install it, and sanity-check with a `DEBUG=True` run before a long run.

## Data

The competition data is **not committed** (it is large and provided by Kaggle).
Download it from the BirdCLEF 2026 competition into `data/`:

```
data/
  train.csv
  taxonomy.csv
  train_soundscapes_labels.csv
  train_audio/          # focal recordings (.ogg)
  train_soundscapes/    # soundscape recordings (.ogg)
```

The agents read these paths directly; `eda.ipynb` analyses them.

## Kaggle submission

1. Upload the chosen `best_model.keras` to Kaggle as a dataset named `birdclef-model`
2. Open `submission.ipynb`, attach the dataset via **Add Data → Your Datasets**
3. Run all cells → `submission.csv` (the notebook auto-discovers the model and smoke-tests it)

## Repository layout

| Path | Purpose |
|------|---------|
| `run.sh` | One-command setup + full run of both backbones |
| `agent.py` | EfficientNet agent loop |
| `Yamnet runs/agent.py` | YAMNet agent loop (bundled offline SavedModel) |
| `prompt_builder.py` | Builds LLM prompts from experiment history |
| `experiment_log.py` | Append-only JSONL experiment logger |
| `experiments.jsonl` | All experiment results (auto-generated) |
| `eda.ipynb` | Exploratory data analysis |
| `results.ipynb` | Leaderboard, backbone comparison, training curves, crash analysis |
| `submission.ipynb` | Kaggle inference notebook |

## Limitations & what's missing

- **Domain-shift gap.** Offline validation (~0.93) is far above the public
  leaderboard (0.71–0.745). The soundscape-validation regime narrows the gap but
  does not close it; clean focal clips remain a poor proxy for noisy multi-species
  soundscapes.
- **YAMNet × soundscape-validation never worked.** That combination consistently
  errored and produced no valid submission, so YAMNet runs with `SOUNDSCAPE_VAL=False`.
- **Incomplete YAMNet logs.** `Yamnet runs/experiments.jsonl` was reset, so the
  per-epoch history of the best YAMNet runs is no longer available — only the
  training-curve images saved at run time survive (used in `results.ipynb`).
- **Two partially-duplicated codebases.** EfficientNet and YAMNet have separate
  `agent.py` / `prompt_builder.py` / `experiment_log.py` rather than one
  parameterised pipeline.
- **No ensembling.** The two backbones are submitted independently; blending them
  (a likely score gain) has not been done.
- **CPU-bound, long runs.** A full run takes hours per backbone; GPU acceleration
  is optional and off by default.
- **Rare-species tail.** Macro-AUC weights all 234 species equally, but many have
  very few training clips — the long tail is under-served (quantified in `eda.ipynb`).

## Future work

Unify the two backbones into one configurable pipeline; ensemble EfficientNet and
YAMNet; per-species threshold/temperature calibration on held-out soundscapes; and
broader test-time augmentation aimed squarely at the domain shift.
