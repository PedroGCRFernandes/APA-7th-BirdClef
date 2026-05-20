"""
BirdCLEF 2026 — Autonomous Research Agent

Usage:
    python agent.py

The agent loads data once, then runs N_ITERATIONS of:
    LLM proposes model head → train → evaluate → log → repeat
"""

import gc
import os
import ast
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import ollama
from sklearn.model_selection import train_test_split

from experiment_log import new_run_id, add_experiment, print_summary, load_successful
from prompt_builder import build_prompt

# ── Version guard ──────────────────────────────────────────────────────────────
# Kaggle's BirdCLEF 2026 image runs these EXACT versions. A newer local Keras
# writes config keys (renorm, quantization_config, ...) that Kaggle's older Keras
# cannot load, so a model trained on the wrong versions silently fails on Kaggle.
KAGGLE_TF    = "2.19.0"
KAGGLE_KERAS = "3.10.0"


def _check_versions():
    tf_v    = tf.__version__
    keras_v = getattr(tf.keras, "__version__", "unknown")
    if (tf_v, keras_v) != (KAGGLE_TF, KAGGLE_KERAS):
        print("\n" + "!" * 72)
        print("VERSION MISMATCH — models you train will NOT load on Kaggle.")
        print(f"  yours : tensorflow {tf_v} / keras {keras_v}")
        print(f"  Kaggle: tensorflow {KAGGLE_TF} / keras {KAGGLE_KERAS}")
        print("  Fix:  pip install -r requirements.txt   (inside the keras_env)")
        print("  Override (not recommended):  ALLOW_VERSION_MISMATCH=1 python agent.py")
        print("!" * 72 + "\n")
        if os.environ.get("ALLOW_VERSION_MISMATCH") != "1":
            raise SystemExit("Aborting — wrong tensorflow/keras versions (see above).")
        return False
    print(f"Version check OK — tensorflow {tf_v} / keras {keras_v} (matches Kaggle)")
    return True


# ── Config ─────────────────────────────────────────────────────────────────────

BASE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

# Audio — YAMNet requires 16 kHz mono
SAMPLE_RATE = 16000
DURATION    = 5

# Training
BATCH_SIZE = 32
N_EPOCHS   = 40

# Acoustic augmentation: per clip, precompute 1 clean + N_AUG_VARIANTS
# waveform-augmented YAMNet embeddings. Training picks a random variant per
# sample each epoch; validation always uses the clean variant. Variants are
# cached, so training stays fast — only the one-time precompute grows ~(1+N)x.
N_AUG_VARIANTS = 6

# Embedding-level masking (SpecAugment analogue): during training, zero a
# contiguous EMB_MASK_WIDTH-wide band of the 1024-d YAMNet vector. Forces the
# head to spread its decision across many feature channels.
EMB_MASK_PROB  = 0.5
EMB_MASK_WIDTH = 128

# Agent
LLM_MODEL    = "gemma4:e4b"  #qwen3.5:9b
N_ITERATIONS = 20
MAX_FIX_RETRIES = 3  # how many times the LLM can try to fix a crash before giving up

# Hardware-adaptive parallelism — uses all but 2 cores (reserved for TF/Metal threads).
# Falls back to 1 on single-core or unknown hardware (== Keras defaults, so a no-op).
_NCPU         = os.cpu_count() or 1
_DATA_WORKERS = max(1, _NCPU - 2)

# Keras 3 takes these on the PyDataset (generator) constructor, NOT on model.fit().
# use_multiprocessing=False keeps thread workers so the shared spectrogram cache
# is not copied per-process (that would exhaust RAM on a large cache).
_LOADER_KW = dict(workers=_DATA_WORKERS, use_multiprocessing=False)

# Apple-Silicon Metal (tensorflow-metal) has pathologically slow kernels for
# EfficientNet (depthwise conv + swish) — often far slower than CPU. Default to
# CPU. Flip to True only to experiment with Metal on a different backbone.
USE_GPU = False

# Debug — set to True to use a tiny data slice for quick pipeline checks
DEBUG = False
DEBUG_SAMPLES = 64  # must be >= BATCH_SIZE

# Validate on held-out soundscapes instead of clean clips, and mix soundscapes
# into training from epoch 1. Aligns val_auc / early-stopping / best-model
# selection with the real Kaggle objective (noisy soundscapes), at the cost of
# a smaller, noisier val set.
#
# Disabled by default: the soundscape val split is only ~hundreds of windows, so
# most of the 234 species have ZERO positives in val. Keras's multi_label AUC
# averages per-class scores across all classes — absent classes contribute ~0
# and drag macro-AUC well below random (we saw 0.26 in run_20260520_090339).
# Falling back to clip-val + plateau-gated Phase 2 on soundscapes restores a
# trustworthy 0.75-0.90 val_auc band.
SOUNDSCAPE_VAL     = False
SOUNDSCAPE_VAL_FRAC = 0.2

# Backbone: YAMNet (Google AudioSet) used as a FROZEN feature extractor.
# Embeddings are precomputed once into the cache; the LLM agent only ever
# designs a head on top of the 1024-d YAMNet embedding. The SavedModel is
# bundled at models/yamnet_savedmodel/ so it loads offline (incl. Kaggle).
YAMNET_PATH = BASE_PATH + "models/yamnet_savedmodel"
EMBED_DIM   = 1024
BACKBONE_NAME = "YAMNet"

# ── Data ───────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(BASE_PATH + "data/train.csv")
    df = df[df["rating"] > 0].reset_index(drop=True)
    df = df[["filename", "primary_label", "secondary_labels"]]
    df["filepath"] = BASE_PATH + "data/train_audio/" + df["filename"]
    print(f"Recordings after filter: {len(df)}")
    return df


def load_taxonomy():
    taxonomy = pd.read_csv(BASE_PATH + "data/taxonomy.csv")
    num_classes  = len(taxonomy)
    # Force string keys: taxonomy primary_label is read as object/str by pandas
    # (mixed numeric/non-numeric values). Normalising here keeps every lookup
    # (clips AND soundscapes) on the same key type regardless of CSV dtype.
    label_to_idx = {str(label).strip(): idx
                    for idx, label in enumerate(taxonomy["primary_label"])}
    print(f"Species (submission columns): {num_classes}")
    return num_classes, label_to_idx


def make_splits(df):
    counts      = df["primary_label"].value_counts()
    rare_labels = counts[counts < 2].index
    rare_df     = df[df["primary_label"].isin(rare_labels)]
    common_df   = df[~df["primary_label"].isin(rare_labels)]

    train_df, val_df = train_test_split(
        common_df, test_size=0.2, random_state=42, stratify=common_df["primary_label"]
    )
    train_df = pd.concat([train_df, rare_df]).reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df


def oversample_rare(train_df, min_count=50):
    """Repeat rows for underrepresented species until each has at least min_count samples."""
    parts = []
    for _, group in train_df.groupby("primary_label"):
        if len(group) < min_count:
            repeats = int(np.ceil(min_count / len(group)))
            group = pd.concat([group] * repeats, ignore_index=True).iloc[:min_count]
        parts.append(group)
    result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"After oversampling: {len(result)} train samples (was {len(train_df)})")
    return result

# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_audio(filepath, offset=0.0):
    """Load DURATION seconds of mono 16 kHz audio (zero-padded if short)."""
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION, offset=offset)
    target = SAMPLE_RATE * DURATION
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio.astype(np.float32)


_yamnet = None

def _get_yamnet():
    """Load the bundled YAMNet SavedModel once (frozen feature extractor)."""
    global _yamnet
    if _yamnet is None:
        _yamnet = tf.saved_model.load(YAMNET_PATH)
    return _yamnet


_N_FRAMES = None  # discovered lazily on first YAMNet call


def _probe_frame_count():
    """Run YAMNet once on a 5-s silence clip to discover its frame count.
    The number depends on YAMNet's internal hop/window and never changes for
    a fixed input length, so we cache it after the first call."""
    global _N_FRAMES
    if _N_FRAMES is None:
        probe = np.zeros(SAMPLE_RATE * DURATION, dtype=np.float32)
        _, frames, _ = _get_yamnet()(probe)
        _N_FRAMES = int(frames.shape[0])
        print(f"YAMNet frame layout: {_N_FRAMES} frames × {EMBED_DIM}-d per {DURATION}-s clip")
    return _N_FRAMES


def audio_to_embedding(audio):
    """Run YAMNet and return its full frame-level output (N_FRAMES, 1024).
    The head is free to mean-pool, attend, or process temporally as it likes
    — switching from mean-pooled (1024,) recovers temporal resolution YAMNet
    actually computes but we were previously throwing away."""
    n = _probe_frame_count()
    _, embeddings, _ = _get_yamnet()(audio)
    emb = embeddings.numpy().astype(np.float32)
    # Defensive pad/truncate — fixed-length input means this almost never fires.
    if emb.shape[0] < n:
        emb = np.pad(emb, ((0, n - emb.shape[0]), (0, 0)))
    elif emb.shape[0] > n:
        emb = emb[:n]
    return emb


def augment_waveform(audio):
    """Acoustic augmentation applied to the raw 16 kHz waveform before YAMNet:
    pitch shift, time stretch, Gaussian field-noise, random gain, time shift.
    Pitch/stretch are applied stochastically (~30% each) so variants stay
    diverse rather than all sounding like the same processing chain."""
    target = SAMPLE_RATE * DURATION

    if np.random.rand() < 0.3:
        try:
            audio = librosa.effects.pitch_shift(
                audio, sr=SAMPLE_RATE, n_steps=float(np.random.uniform(-2.0, 2.0))
            )
        except Exception:
            pass

    if np.random.rand() < 0.3:
        try:
            audio = librosa.effects.time_stretch(audio, rate=float(np.random.uniform(0.9, 1.1)))
            if len(audio) > target:
                audio = audio[:target]
            elif len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))
        except Exception:
            pass

    audio = audio + np.random.randn(len(audio)).astype(np.float32) * np.random.uniform(0.001, 0.015)
    audio = audio * np.random.uniform(0.7, 1.3)
    shift = np.random.randint(-SAMPLE_RATE // 2, SAMPLE_RATE // 2)
    audio = np.roll(audio, shift)
    return audio.astype(np.float32)


# ── Embedding cache ────────────────────────────────────────────────────────────

def precompute_embeddings(df, soundscape_df=None):
    """Decode audio in parallel, then run YAMNet once per clip into RAM.

    Audio decode is parallelised (librosa releases the GIL); YAMNet inference
    is kept on the main thread since the SavedModel is not thread-safe and
    decode is the parallelisable part anyway.
    """
    cache = {}

    filepaths = df["filepath"].unique()
    print(f"Pre-computing {len(filepaths)} clips x {1 + N_AUG_VARIANTS} variants "
          f"(1 clean + {N_AUG_VARIANTS} acoustic) ({_DATA_WORKERS} decode worker(s))...")

    def _decode_clip(fp):
        try:
            return fp, load_audio(fp)
        except Exception:
            return fp, None

    with ThreadPoolExecutor(max_workers=_DATA_WORKERS) as pool:
        for i, (fp, wav) in enumerate(pool.map(_decode_clip, filepaths)):
            if wav is not None:
                variants = [audio_to_embedding(wav)]  # index 0 = clean (used for val)
                for _ in range(N_AUG_VARIANTS):
                    variants.append(audio_to_embedding(augment_waveform(wav)))
                cache[fp] = np.stack(variants).astype(np.float32)  # (1+K, N_FRAMES, 1024)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(filepaths)} done")

    if soundscape_df is not None:
        print(f"Pre-computing {len(soundscape_df)} soundscape embeddings ({_DATA_WORKERS} decode worker(s))...")
        rows_list = list(soundscape_df.iterrows())

        def _decode_soundscape(item):
            _, row = item
            key = (row["filepath"], row["start_sec"])
            try:
                return key, load_audio(row["filepath"], offset=row["start_sec"])
            except Exception:
                return key, None

        with ThreadPoolExecutor(max_workers=_DATA_WORKERS) as pool:
            for key, wav in pool.map(_decode_soundscape, rows_list):
                if wav is not None:
                    cache[key] = audio_to_embedding(wav)

    total_mb = sum(v.nbytes for v in cache.values()) / 1024 ** 2
    print(f"Cache ready: {len(cache)} embeddings, ~{total_mb:.0f} MB")
    total_gb = total_mb / 1024
    if total_gb > 12:
        print("WARNING: embedding cache is above 12 GB. You may hit swap; consider lowering N_AUG_VARIANTS.")
    return cache


def encode_labels(primary_label, secondary_labels_str, label_to_idx, num_classes):
    vec = np.zeros(num_classes, dtype=np.float32)
    key = str(primary_label).strip()
    if key in label_to_idx:
        vec[label_to_idx[key]] = 1.0
    try:
        for sec in ast.literal_eval(secondary_labels_str):
            sk = str(sec).strip()
            if sk in label_to_idx:
                vec[label_to_idx[sk]] = 1.0
    except (ValueError, SyntaxError):
        pass
    return vec

# ── Data generator ─────────────────────────────────────────────────────────────

class BirdDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataframe, label_to_idx, num_classes,
                 batch_size=BATCH_SIZE, augment=False, shuffle=True, cache=None, **kwargs):
        super().__init__(**kwargs)
        self.df           = dataframe.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.num_classes  = num_classes
        self.batch_size   = batch_size
        self.augment      = augment
        self.shuffle      = shuffle
        self.cache        = cache
        self.on_epoch_end()

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch.iterrows():
            if self.cache is not None and row["filepath"] in self.cache:
                emb = self._pick_variant(self.cache[row["filepath"]])
            else:
                emb = audio_to_embedding(load_audio(row["filepath"]))
            X.append(emb)
            y.append(encode_labels(
                row["primary_label"], row["secondary_labels"],
                self.label_to_idx, self.num_classes
            ))
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        if self.augment:
            X = self._band_mask(X)
            X, y = self._mixup(X, y)
        return X, y

    def _band_mask(self, X):
        """SpecAugment-style masking on the FEATURE axis of the frame-level
        YAMNet output (shape: batch × N_FRAMES × 1024). For each sample, with
        probability EMB_MASK_PROB, zero out a contiguous EMB_MASK_WIDTH-wide
        band of channels across all frames. Forces the head to spread its
        decision across many feature channels."""
        d = X.shape[-1]
        for i in range(X.shape[0]):
            if np.random.rand() < EMB_MASK_PROB:
                start = np.random.randint(0, d - EMB_MASK_WIDTH + 1)
                X[i, ..., start:start + EMB_MASK_WIDTH] = 0.0
        return X

    def _pick_variant(self, emb):
        """Cached clips hold (1+K, N_FRAMES, 1024): a clean variant at index 0 plus K
        acoustic ones. Train picks a random variant; val always uses clean."""
        if emb.ndim == 3:
            i = np.random.randint(emb.shape[0]) if self.augment else 0
            return emb[i]
        return emb

    def _mixup(self, X, y):
        """Batch-level mixup — blend each sample with a random partner, mixing
        embeddings and labels. Complements the acoustic-variant augmentation.
        Broadcasts to either (B, D) (mean-pooled) or (B, T, D) (frame-level)."""
        n = X.shape[0]
        lam = np.random.beta(0.4, 0.4, size=n).astype(np.float32)
        lam_X = lam.reshape((n,) + (1,) * (X.ndim - 1))
        lam_y = lam.reshape((n,) + (1,) * (y.ndim - 1))
        perm = np.random.permutation(n)
        X = lam_X * X + (1.0 - lam_X) * X[perm]
        y = lam_y * y + (1.0 - lam_y) * y[perm]
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)


# ── Soundscape data ─────────────────────────────────────────────────────────────

def load_soundscape_data(label_to_idx, num_classes):
    labels_path  = BASE_PATH + "data/train_soundscapes_labels.csv"
    soundscape_dir = BASE_PATH + "data/train_soundscapes/"

    if not os.path.exists(labels_path):
        print("No soundscape labels found — skipping phase 2.")
        return None

    df = pd.read_csv(labels_path)
    rows = []
    for _, row in df.iterrows():
        h, m, s = str(row["start"]).split(":")
        start_sec = int(h) * 3600 + int(m) * 60 + int(s)

        vec = np.zeros(num_classes, dtype=np.float32)
        for taxon_id in str(row["primary_label"]).split(";"):
            # NOTE: int() cast — label_to_idx keys are strings, so this fails
            # for most rows and leaves the target vector all-zero (~60% of
            # soundscape windows). Restored to match the original baseline.
            try:
                key = int(taxon_id)
            except ValueError:
                continue
            if key in label_to_idx:
                vec[label_to_idx[key]] = 1.0

        filepath = soundscape_dir + row["filename"]
        if os.path.exists(filepath):
            rows.append({"filepath": filepath, "start_sec": start_sec, "label_vector": vec})

    result = pd.DataFrame(rows)
    # NOTE: no zero-vector filter — keeps the all-zero-target windows produced
    # by the int() cast above. Restored to match the original baseline.
    print(f"Soundscape windows: {len(result)}")
    return result


class SoundscapeGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataframe, batch_size=BATCH_SIZE, shuffle=True, cache=None, **kwargs):
        super().__init__(**kwargs)
        self.df         = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.cache      = cache
        self.on_epoch_end()

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, idx):
        batch = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for _, row in batch.iterrows():
            key = (row["filepath"], row["start_sec"])
            if self.cache is not None and key in self.cache:
                X.append(self.cache[key])
            else:
                X.append(audio_to_embedding(
                    load_audio(row["filepath"], offset=row["start_sec"])
                ))
            y.append(row["label_vector"])
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)


class CombinedGenerator(tf.keras.utils.Sequence):
    """Serves batches from two generators back-to-back within each epoch."""

    def __init__(self, gen_a, gen_b, **kwargs):
        super().__init__(**kwargs)
        self.gen_a = gen_a
        self.gen_b = gen_b

    def __len__(self):
        return len(self.gen_a) + len(self.gen_b)

    def __getitem__(self, idx):
        if idx < len(self.gen_a):
            return self.gen_a[idx]
        return self.gen_b[idx - len(self.gen_a)]

    def on_epoch_end(self):
        self.gen_a.on_epoch_end()
        self.gen_b.on_epoch_end()


class _BestWeightsCallback(tf.keras.callbacks.Callback):
    """Keeps the best val_auc weights in RAM — avoids serialization entirely."""

    def __init__(self):
        super().__init__()
        self.best_weights = None
        self.best_auc     = 0.0

    def on_epoch_end(self, epoch, logs=None):
        auc = (logs or {}).get("val_auc", 0.0)
        if auc > self.best_auc:
            self.best_auc     = auc
            self.best_weights = self.model.get_weights()


class PlateauCallback(tf.keras.callbacks.Callback):
    """Stops training after `patience` epochs without val_auc moving by at
    least `min_delta` (an *absolute* val_auc change, e.g. 0.005).
    """

    def __init__(self, patience=4, min_delta=0.01):
        super().__init__()
        self.patience        = patience
        self.min_delta       = min_delta
        self.best_auc        = 0.0
        self.wait            = 0
        self.plateau_reached = False
        self.epochs_trained  = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_trained = epoch + 1
        current = (logs or {}).get("val_auc", 0.0)
        required = self.min_delta
        # Plateau = val_auc barely moves in EITHER direction. A large move
        # (up OR down) means the model is still changing, so reset patience;
        # only a small absolute change counts toward the plateau.
        if abs(current - self.best_auc) >= required:
            self.wait = 0
        else:
            self.wait += 1
            print(f"    [plateau] val_auc stable for {self.wait}/{self.patience} epochs")
            if self.wait >= self.patience:
                self.plateau_reached = True
                self.model.stop_training = True
                print(f"    [plateau] val_auc plateaued — stopping after {self.epochs_trained} epochs")
        # best_auc always tracks the max seen (never lowered by a drop)
        if current > self.best_auc:
            self.best_auc = current


# ── Backbone ───────────────────────────────────────────────────────────────────

def build_backbone():
    """YAMNet is applied during precompute, so the in-graph 'backbone' is a
    frozen identity over the cached frame-level embeddings: shape (N_FRAMES, 1024).
    The head decides how to collapse the time axis — mean pool, attention,
    Conv1D, learnable pooling, whatever it likes."""
    n_frames = _probe_frame_count()
    inputs = tf.keras.Input(shape=(n_frames, EMBED_DIM))
    outputs = tf.keras.layers.Activation("linear", name="yamnet_embedding")(inputs)
    backbone_model = tf.keras.Model(inputs, outputs, name="yamnet_passthrough")
    backbone_model.trainable = False
    print(f"Backbone: {BACKBONE_NAME} (frozen) | input shape ({n_frames}, {EMBED_DIM})")
    return backbone_model, (n_frames, EMBED_DIM)

# ── LLM helpers ────────────────────────────────────────────────────────────────

def call_llm(prompt, retries=3, wait=10):
    for attempt in range(retries):
        try:
            response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"  LLM call failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def extract_code(llm_response):
    match = re.search(r"```python\s*(.*?)```", llm_response, re.DOTALL)
    if not match:
        return None
    # dedent first so a uniformly-indented snippet doesn't crash exec on line 1,
    # then strip trailing whitespace. Leading newlines are fine for exec.
    return textwrap.dedent(match.group(1)).strip()


def ask_llm_to_analyze(result, code, iteration, history):
    """Call the LLM to reflect on a completed training run and suggest what to change."""
    best = max(
        (e for e in history if e.get("val_auc") is not None),
        key=lambda e: e["val_auc"], default=None
    )
    best_line = f"Best across all runs so far: val_auc={best['val_auc']:.4f}" if best else "This is the first successful run."

    val_auc_str  = f"{result['val_auc']:.4f}"  if result['val_auc']  is not None else 'N/A'
    val_loss_str = f"{result['val_loss']:.4f}" if result['val_loss'] is not None else 'N/A'

    prompt = f"""You are reviewing the results of a bird species audio classification experiment.

EXPERIMENT {iteration} RESULTS
-------------------------------
Status       : {result['status']}
val_auc      : {val_auc_str}
val_loss     : {val_loss_str}
epochs run   : {result['epochs_trained']}
training time: {result['training_time_sec']}s
{best_line}

CODE USED:
```python
{code}
```

In 2-3 sentences: what does this result tell you? What worked or didn't, and what specific change would most likely improve val_auc in the next iteration?
"""
    try:
        return call_llm(prompt).strip()
    except Exception:
        return ""


def ask_llm_to_fix(code, error, attempt, input_shape):
    return call_llm(f"""The Keras model head you generated crashed with this error:

ERROR:
{error}

FAILED CODE:
```python
{code}
```

Fix attempt {attempt}/{MAX_FIX_RETRIES}. Rules reminder:
- Input shape is {input_shape} — {input_shape[0]} time frames × {input_shape[1]}-d embedding
- Your code MUST start with exactly these two lines:
      inputs = tf.keras.Input(shape={input_shape})
      x      = backbone_model(inputs, training=False)
- `backbone_model` is a Model object: NEVER pass it as a positional argument
  to a layer call or to `tf.keras.Model(...)`. Only tensors may be passed
  positionally to layers.
- Shape rules for merges:
  * `Add()([a, b])` requires IDENTICAL shapes. To add a Dense(N) branch to the
    1024-d YAMNet output, project BOTH sides to N (or drop the residual).
  * `Concatenate()` requires matching shapes on every axis except the concat axis.
- Do not define any classes
- Available layers: Dense, Dropout, BatchNormalization, LayerNormalization, Activation,
  Conv1D, Conv2D, DepthwiseConv2D, SeparableConv2D,
  MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D,
  GlobalMaxPooling1D, GlobalMaxPooling2D, Flatten, Reshape, Lambda,
  Concatenate, Add, Multiply, MultiHeadAttention, Attention, Input, Model
- No model.compile() or model.fit()
- Final variable must be named `model`, ending with `model = Model(inputs, outputs)`
  where `outputs = Dense(234, activation='sigmoid')(x)`

Return only the fixed Python code in a ```python``` block. No explanations.
""")


def strip_forbidden_calls(code):
    forbidden = [
        "model.fit(", "train_generator", "val_generator", "model.compile(",
    ]
    lines = []
    for line in code.splitlines():
        if any(term in line for term in forbidden):
            lines.append(f"# [removed]: {line}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _auc_sanity_warnings(val_aucs):
    if not val_aucs:
        return
    first = val_aucs[0]
    last = val_aucs[-1]

    if first >= 0.99:
        print(
            "WARNING: epoch-1 val_auc is extremely high (>=0.99). This usually indicates "
            "micro-AUC or a label/validation leakage. Double-check label mapping and the "
            "train/val split."
        )
    elif first < 0.75:
        print(
            "WARNING: epoch-1 val_auc is below the expected 0.75-0.90 band. Consider checking "
            "taxonomy/label mapping, soundscape validation setup, or increasing head capacity."
        )

    if last < 0.70:
        print(
            "NOTE: val_auc remains low. Try focal loss, a larger head, or stronger augmentation to improve AUC."
        )


def execute_safely(code, backbone_model, train_generator, val_generator, soundscape_generator=None):
    if code is None:
        return _crashed("No code block found in LLM response")

    if "backbone_model" not in code:
        return _crashed("LLM code does not use backbone_model")

    if re.search(r"^\s*class\s+", code, re.MULTILINE):
        return _crashed("LLM defined a custom class — only use backbone_model and allowed layers")

    try:
        ns = {
            "tf"              : tf,
            "np"              : np,
            "backbone_model"  : backbone_model,
            # Dense / regularisation
            "Dense"                : tf.keras.layers.Dense,
            "Dropout"              : tf.keras.layers.Dropout,
            "BatchNormalization"   : tf.keras.layers.BatchNormalization,
            "LayerNormalization"   : tf.keras.layers.LayerNormalization,
            "Activation"           : tf.keras.layers.Activation,
            # Convolution
            "Conv1D"               : tf.keras.layers.Conv1D,
            "Conv2D"               : tf.keras.layers.Conv2D,
            "DepthwiseConv2D"      : tf.keras.layers.DepthwiseConv2D,
            "SeparableConv2D"      : tf.keras.layers.SeparableConv2D,
            # Pooling
            "MaxPooling1D"         : tf.keras.layers.MaxPooling1D,
            "MaxPooling2D"         : tf.keras.layers.MaxPooling2D,
            "AveragePooling1D"     : tf.keras.layers.AveragePooling1D,
            "AveragePooling2D"     : tf.keras.layers.AveragePooling2D,
            "GlobalAveragePooling1D": tf.keras.layers.GlobalAveragePooling1D,
            "GlobalAveragePooling2D": tf.keras.layers.GlobalAveragePooling2D,
            "GlobalMaxPooling1D"   : tf.keras.layers.GlobalMaxPooling1D,
            "GlobalMaxPooling2D"   : tf.keras.layers.GlobalMaxPooling2D,
            # Shape / merge
            "Flatten"              : tf.keras.layers.Flatten,
            "Reshape"              : tf.keras.layers.Reshape,
            "Lambda"               : tf.keras.layers.Lambda,
            "Concatenate"          : tf.keras.layers.Concatenate,
            "Add"                  : tf.keras.layers.Add,
            "Multiply"             : tf.keras.layers.Multiply,
            # Attention
            "MultiHeadAttention"   : tf.keras.layers.MultiHeadAttention,
            "Attention"            : tf.keras.layers.Attention,
            # Model building
            "Input"                : tf.keras.Input,
            "Model"                : tf.keras.Model,
            # allow standard imports (math, random, etc.) inside exec
            "__builtins__"         : __builtins__,
        }
        clean_code = strip_forbidden_calls(code)

        start = time.time()
        exec(clean_code, ns)  # noqa: S102
        model = ns.get("model")

        if model is None:
            return _crashed("LLM code did not assign a variable named 'model'")

        lr   = ns.get("learning_rate", 1e-3)
        # Default loss: BCE with light label smoothing — multi-label bird audio
        # has noisy secondaries (overlapping calls, mislabelled background),
        # 0.02 smoothing improves calibration without hurting macro-AUROC.
        loss = ns.get("loss", tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02))

        # NOTE: micro-AUROC (Keras default) — flattens (samples × classes), so
        # rows with 1–2 positives out of 234 score trivially high (~0.99).
        # Restored to match the original micro-AUC run baselines on request.
        num_out = model.output_shape[-1]

        # Cosine LR decay over the full Phase-1 budget. The LLM still picks
        # `learning_rate` as the *initial* LR; cosine takes over from there and
        # smoothly anneals to ~0 by the last epoch. Pairs with the longer
        # N_EPOCHS budget so late epochs do fine-tuning rather than thrashing.
        steps_per_epoch = max(1, len(train_generator))
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=steps_per_epoch * N_EPOCHS,
            alpha=0.01,  # floor at 1% of initial — keep tiny updates alive
        )
        # AdamW with small weight decay — slightly better generalisation than
        # plain Adam on small heads sitting on a frozen backbone. Pairs well
        # with cosine decay: as LR shrinks, decay still pulls weights toward 0.
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
            loss=loss,
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )

        # ── Phase 1: main training ──────────────────────────────────────────
        # SOUNDSCAPE_VAL=True  → soundscape_generator is None: single phase,
        #   train_generator already mixes clips + soundscapes, val = soundscapes.
        # SOUNDSCAPE_VAL=False → soundscape_generator is set: classic two-phase
        #   (clips first, then clips+soundscapes after a plateau).
        # patience 5: soundscape-val is small/noisy, so tolerate more
        # epoch-to-epoch wobble before declaring a plateau
        plateau_cb    = PlateauCallback(patience=5, min_delta=0.005)
        best_wts_cb   = _BestWeightsCallback()
        history = model.fit(
            train_generator, validation_data=val_generator,
            epochs=N_EPOCHS, verbose=1, callbacks=[plateau_cb, best_wts_cb],
        )

        # best (auc, weights) seen so far across all phases
        best_auc     = best_wts_cb.best_auc
        best_weights = best_wts_cb.best_weights

        # ── Phase 2 (only when SOUNDSCAPE_VAL=False): plateau-gated soundscapes ──
        remaining = N_EPOCHS - plateau_cb.epochs_trained
        if plateau_cb.plateau_reached and soundscape_generator is not None and remaining > 0:
            print(f"  Phase 2: {remaining} epoch(s) on train_audio + soundscape combined...")
            combined_generator = CombinedGenerator(train_generator, soundscape_generator, **_LOADER_KW)
            plateau_cb2  = PlateauCallback(patience=3, min_delta=0.01)
            best_wts_cb2 = _BestWeightsCallback()
            h2 = model.fit(
                combined_generator, validation_data=val_generator,
                epochs=remaining, verbose=1, callbacks=[plateau_cb2, best_wts_cb2],
            )
            history.history["val_auc"].extend(h2.history["val_auc"])
            history.history["val_loss"].extend(h2.history["val_loss"])
            history.history["loss"].extend(h2.history["loss"])
            history.history["auc"].extend(h2.history.get("auc", []))
            if best_wts_cb2.best_auc > best_auc:
                best_auc     = best_wts_cb2.best_auc
                best_weights = best_wts_cb2.best_weights

        elapsed        = round(time.time() - start)
        val_auc        = history.history["val_auc"][-1]
        val_loss       = history.history["val_loss"][-1]
        epochs_trained = len(history.history["val_auc"])
        # restore the best epoch's weights if the final epoch was worse
        if val_auc < best_auc and best_weights is not None:
            model.set_weights(best_weights)
            val_auc = best_auc
            print(f"  Restored best-epoch weights ({val_auc:.4f})")
        print(f"  train_loss={history.history['loss'][-1]:.4f} | val_loss={val_loss:.4f} | val_auc={val_auc:.4f} | epochs={epochs_trained}")

        _auc_sanity_warnings(history.history.get("val_auc", []))

        epoch_history = {
            "loss":     history.history.get("loss", []),
            "val_loss": history.history.get("val_loss", []),
            "auc":      history.history.get("auc", []),
            "val_auc":  history.history.get("val_auc", []),
        }
        return {
            "status": "success", "error": None,
            "val_auc": val_auc, "val_loss": val_loss,
            "training_time_sec": elapsed, "epochs_trained": epochs_trained,
            "model": model, "epoch_history": epoch_history,
        }

    except Exception as e:
        return _crashed(str(e))


def _crashed(msg):
    return {
        "status": "crashed", "error": msg,
        "val_auc": None, "val_loss": None,
        "training_time_sec": None, "epochs_trained": 0,
    }

# ── Model saving ──────────────────────────────────────────────────────────────

def save_keras_clean(model, path):
    """Save a .keras model, then strip the dead BatchNormalization `renorm*`
    kwargs from its config. Keras 3.14 still writes those, but Kaggle's stricter
    Keras rejects them on load — stripping here makes every saved model load on
    Kaggle without any notebook-side shim."""
    model.save(path)

    DEAD = ("renorm", "renorm_clipping", "renorm_momentum")

    def _strip(obj):
        if isinstance(obj, dict):
            for k in DEAD:
                obj.pop(k, None)
            for v in obj.values():
                _strip(v)
        elif isinstance(obj, list):
            for v in obj:
                _strip(v)

    import zipfile, json
    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        data  = {n: zin.read(n) for n in names}

    cfg = json.loads(data["config.json"])
    _strip(cfg)
    data["config.json"] = json.dumps(cfg).encode()

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, data[n])


# ── Run report ────────────────────────────────────────────────────────────────

def generate_run_report(run_id, history):
    report_dir = BASE_PATH + f"runs/{run_id}/"
    # folder already created at run start; models already saved there during training

    # Save each iteration's code
    for exp in history:
        if exp.get("code"):
            fname = f"iter{exp['iteration']:02d}_{exp['label']}.py"
            with open(report_dir + fname, "w") as f:
                f.write(exp["code"])

    # Plot loss and AUC curves
    successful = [e for e in history if e.get("epoch_history", {}).get("val_loss")]
    if successful:
        fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Training curves — {run_id}", fontsize=11)

        for exp in successful:
            eh     = exp["epoch_history"]
            label  = exp["label"]
            auc_str = f" ({exp['val_auc']:.4f})" if exp.get("val_auc") else ""
            epochs = range(1, len(eh["val_loss"]) + 1)
            ax_loss.plot(epochs, eh["val_loss"], label=label + auc_str)
            ax_auc.plot(epochs, eh["val_auc"],  label=label + auc_str)

        for ax, title, ylabel in [
            (ax_loss, "Validation Loss",  "Loss"),
            (ax_auc,  "Validation AUC",   "AUC"),
        ]:
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(report_dir + "training_curves.png", dpi=150)
        plt.close()

    print(f"\nRun report saved → {report_dir}")
    print(f"  Code files  : {len([e for e in history if e.get('code')])} iterations")
    print(f"  Model files : best_model.keras + best_model_weights.weights.h5")
    print(f"  Plot        : training_curves.png")


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(backbone_model, feature_dim, train_generator, val_generator, soundscape_generator=None):
    run_id = new_run_id()
    print(f"\nAgent run: {run_id}")
    print(f"LLM={LLM_MODEL} | Backbone={BACKBONE_NAME} | Iterations={N_ITERATIONS}")
    print("-" * 60)

    run_dir    = BASE_PATH + f"runs/{run_id}/"
    models_dir = BASE_PATH + "models/"
    os.makedirs(run_dir,    exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    best_auc = 0.0  # best within this run (controls the per-run folder copy)

    # best val_auc ever recorded — protects models/ (the Kaggle copy) from being
    # clobbered by a weaker run, since best_auc resets to 0 every run
    prev = [e["val_auc"] for e in load_successful() if e.get("val_auc") is not None]
    global_best_auc = max(prev) if prev else 0.0
    print(f"Global best val_auc on record: {global_best_auc:.4f} "
          f"(models/ only overwritten if a run beats this)")

    history = []  # grows each iteration — fed into the next prompt

    for iteration in range(1, N_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"[Iteration {iteration}/{N_ITERATIONS}]")

        # ── Build prompt with accumulated history ───────────────────────────
        _fdim = int(backbone_model.output_shape[-1])
        _nfr  = int(backbone_model.output_shape[-2])
        prompt = build_prompt(_fdim, backbone_name=BACKBONE_NAME,
                              n_frames=_nfr, history=history)

        if history:
            best = max((e for e in history if e["val_auc"] is not None), key=lambda e: e["val_auc"], default=None)
            print(f"  Feedback: {len(history)} past experiment(s) sent to LLM", end="")
            print(f" | best so far: val_auc={best['val_auc']:.4f}" if best else "")
        else:
            print("  Feedback: none (first iteration — no history yet)")

        # ── LLM proposes architecture ───────────────────────────────────────
        print("  Calling LLM...")
        try:
            llm_response = call_llm(prompt)
        except Exception as e:
            print(f"  LLM unavailable after retries: {e} — skipping iteration {iteration}")
            if not DEBUG:
                add_experiment(
                    run_id=run_id, iteration=iteration,
                    label=f"yamnet_iter{iteration}", architecture="", code="",
                    status="crashed", crash_count=0, error=f"LLM unavailable: {e}",
                    val_auc=None, val_loss=None, epochs_trained=0,
                    training_time_sec=None, llm_analysis="", epoch_history={},
                    notes=f"backbone={BACKBONE_NAME}, epochs={N_EPOCHS}",
                )
            gc.collect()
            continue
        code = extract_code(llm_response)

        # ── Train ───────────────────────────────────────────────────────────
        print("  Running generated code...")
        result = execute_safely(code, backbone_model, train_generator, val_generator, soundscape_generator)
        crash_count = 0

        # retry loop — LLM gets MAX_FIX_RETRIES attempts to fix a crash
        while result["status"] == "crashed" and crash_count < MAX_FIX_RETRIES:
            crash_count += 1
            print(f"  CRASHED (attempt {crash_count}/{MAX_FIX_RETRIES}): {result['error']}")
            print(f"  Asking LLM to fix...")
            try:
                fix_response = ask_llm_to_fix(
                    code, result["error"], crash_count,
                    input_shape=(_nfr, _fdim),
                )
            except Exception as e:
                print(f"  Fix request failed ({e}) — keeping previous code")
                fix_response = ""
            fixed_code = extract_code(fix_response)
            if fixed_code:
                code = fixed_code
            result = execute_safely(code, backbone_model, train_generator, val_generator, soundscape_generator)

        if result["status"] == "crashed":
            print(f"  FAILED after {crash_count} fix attempt(s): {result['error']}")
            llm_analysis = ""
        else:
            print(f"  SUCCESS after {crash_count} crash(es)")
            if result["val_auc"] > best_auc:
                best_auc = result["val_auc"]
                # always save this run's best into its own folder
                save_keras_clean(result["model"], run_dir + "best_model.keras")
                result["model"].save_weights(run_dir + "best_model_weights.weights.h5")
                print(f"  New run-best val_auc={best_auc:.4f} — saved to {run_dir}")

                # only overwrite the global Kaggle copy if it beats every prior run
                if result["val_auc"] > global_best_auc:
                    global_best_auc = result["val_auc"]
                    save_keras_clean(result["model"], models_dir + "best_model.keras")
                    result["model"].save_weights(models_dir + "best_model_weights.weights.h5")
                    print(f"  New GLOBAL best — updated models/ ({global_best_auc:.4f})")

            print("  Asking LLM to analyse results...")
            llm_analysis = ask_llm_to_analyze(result, code, iteration, history)
            print(f"  LLM analysis: {llm_analysis[:200]}")

        # ── Log and update history ──────────────────────────────────────────
        experiment = {
            "iteration"    : iteration,
            "label"        : f"yamnet_iter{iteration}",
            "architecture" : llm_response[:300],
            "code"         : code,
            "status"       : result["status"],
            "error"        : result["error"],
            "val_auc"      : result["val_auc"],
            "val_loss"     : result["val_loss"],
            "epoch_history": result.get("epoch_history", {}),
        }
        history.append(experiment)

        if not DEBUG:
            add_experiment(
                run_id            = run_id,
                iteration         = iteration,
                label             = experiment["label"],
                architecture      = experiment["architecture"],
                code              = code,
                status            = result["status"],
                crash_count       = crash_count,
                error             = result["error"],
                val_auc           = result["val_auc"],
                val_loss          = result["val_loss"],
                epochs_trained    = result.get("epochs_trained", 0),
                training_time_sec = result["training_time_sec"],
                llm_analysis      = llm_analysis,
                epoch_history     = experiment["epoch_history"],
                notes             = f"backbone={BACKBONE_NAME}, epochs={N_EPOCHS}",
            )

        if "model" in result:
            del result["model"]
        gc.collect()

    print("\n" + "=" * 60)
    print("Agent run complete.")
    print_summary(run_id)
    if not DEBUG:
        generate_run_report(run_id, history)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not USE_GPU:
        try:
            tf.config.set_visible_devices([], "GPU")
            print("GPU disabled — training on CPU (set USE_GPU=True in agent.py to try Metal)")
        except RuntimeError as e:
            print(f"Could not disable GPU (already initialized): {e}")
    _check_versions()
    df = load_data()
    num_classes, label_to_idx = load_taxonomy()
    train_df, val_df = make_splits(df)

    backbone_model, feature_dim = build_backbone()

    if DEBUG:
        print(f"[DEBUG] Using {DEBUG_SAMPLES} samples for train and val")
        train_df = train_df.iloc[:DEBUG_SAMPLES]
        val_df   = val_df.iloc[:DEBUG_SAMPLES]
    else:
        train_df = oversample_rare(train_df, min_count=50)

    soundscape_df = load_soundscape_data(label_to_idx, num_classes)
    if DEBUG and soundscape_df is not None:
        # keep enough that the 80/20 split still yields >= 1 batch each side
        soundscape_df = soundscape_df.iloc[:DEBUG_SAMPLES * 4].reset_index(drop=True)

    # YAMNet embeddings are tiny (1024 floats/clip), so precompute always —
    # even in DEBUG — to exercise the real path and keep training instant.
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    emb_cache = precompute_embeddings(all_df, soundscape_df)

    # Decide whether the soundscape-validation regime is usable: need enough
    # soundscape windows to carve a meaningful held-out val split.
    use_ss_val = (
        SOUNDSCAPE_VAL
        and soundscape_df is not None
        # both splits must hold at least one batch
        and len(soundscape_df) * SOUNDSCAPE_VAL_FRAC >= BATCH_SIZE
        and len(soundscape_df) * (1 - SOUNDSCAPE_VAL_FRAC) >= BATCH_SIZE
    )

    clip_train_gen = BirdDataGenerator(
        train_df, label_to_idx, num_classes, augment=True, shuffle=True, cache=emb_cache,
        **_LOADER_KW,
    )

    if use_ss_val:
        # Split soundscapes into train/val; train on clips + soundscape-train
        # from epoch 1, validate on the held-out soundscape-val (the real
        # objective). Phase 2 is disabled (soundscape_generator=None) because
        # soundscapes are already mixed into training throughout.
        ss_train_df, ss_val_df = train_test_split(
            soundscape_df, test_size=SOUNDSCAPE_VAL_FRAC, random_state=42
        )
        ss_train_gen    = SoundscapeGenerator(ss_train_df, shuffle=True,  cache=emb_cache)
        train_generator = CombinedGenerator(clip_train_gen, ss_train_gen, **_LOADER_KW)
        val_generator   = SoundscapeGenerator(ss_val_df, shuffle=False, cache=emb_cache, **_LOADER_KW)
        soundscape_generator = None
        print(f"SOUNDSCAPE_VAL on — soundscape train/val: "
              f"{len(ss_train_df)}/{len(ss_val_df)} windows")
        print(f"Train batches/epoch: {len(train_generator)} "
              f"(clips {len(clip_train_gen)} + soundscape {len(ss_train_gen)})")
        print(f"Val (soundscape) batches/epoch: {len(val_generator)}")
    else:
        # Original regime: clip-val + plateau-gated Phase 2 on soundscapes.
        train_generator = clip_train_gen
        val_generator = BirdDataGenerator(
            val_df, label_to_idx, num_classes, shuffle=False, cache=emb_cache,
            **_LOADER_KW,
        )
        soundscape_generator = None
        if soundscape_df is not None and len(soundscape_df) >= BATCH_SIZE:
            soundscape_generator = SoundscapeGenerator(soundscape_df, shuffle=True, cache=emb_cache, **_LOADER_KW)
            print(f"Soundscape batches/epoch: {len(soundscape_generator)}")
        print(f"Train samples: {len(train_df)} | batches/epoch: {len(train_generator)}")
        print(f"Val   samples: {len(val_df)}   | batches/epoch: {len(val_generator)}")

    X_batch, y_batch = train_generator[0]
    print(f"Sanity check — X: {X_batch.shape} | y: {y_batch.shape}")

    run_agent(backbone_model, feature_dim, train_generator, val_generator, soundscape_generator)
