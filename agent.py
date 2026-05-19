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
import time

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

# Audio
SAMPLE_RATE = 32000
DURATION    = 5

# Mel-spectrogram

N_MELS = 64 #64 - faster training, lower resolution; 128 - slower training, more detail
F_MAX  = 16000

# Training
BATCH_SIZE = 32
N_EPOCHS   = 20

# Agent
LLM_MODEL    = "gemma4:e4b"  #qwen3.5:9b
N_ITERATIONS = 8
MAX_FIX_RETRIES = 3  # how many times the LLM can try to fix a crash before giving up

# Debug — set to True to use a tiny data slice for quick pipeline checks
DEBUG = False
DEBUG_SAMPLES = 64  # must be >= BATCH_SIZE

# Validate on held-out soundscapes instead of clean clips, and mix soundscapes
# into training from epoch 1. Aligns val_auc / early-stopping / best-model
# selection with the real Kaggle objective (noisy soundscapes), at the cost of
# a smaller, noisier val set. Set False to restore clip-val + plateau Phase 2.
SOUNDSCAPE_VAL     = True
SOUNDSCAPE_VAL_FRAC = 0.2

# Backbone: "efficientnet"
BACKBONE  = "efficientnet"
FINE_TUNE = True

BACKBONE_CLASS_NAME = {
    "efficientnet": "EfficientNetB0",
}

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
    label_to_idx = {label: idx for idx, label in enumerate(taxonomy["primary_label"])}
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

# ── Spectrogram cache ──────────────────────────────────────────────────────────

def precompute_spectrograms(df, soundscape_df=None):
    """Pre-load all audio and compute mel-spectrograms once into RAM."""
    cache = {}

    filepaths = df["filepath"].unique()
    print(f"Pre-computing {len(filepaths)} train/val spectrograms...")
    for i, fp in enumerate(filepaths):
        try:
            cache[fp] = audio_to_melspectrogram(load_audio(fp))
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(filepaths)} done")

    if soundscape_df is not None:
        print(f"Pre-computing {len(soundscape_df)} soundscape spectrograms...")
        for _, row in soundscape_df.iterrows():
            key = (row["filepath"], row["start_sec"])
            try:
                audio, _ = librosa.load(
                    row["filepath"], sr=SAMPLE_RATE,
                    duration=DURATION, offset=row["start_sec"]
                )
                if len(audio) < SAMPLE_RATE * DURATION:
                    audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
                cache[key] = audio_to_melspectrogram(audio)
            except Exception:
                pass

    total_mb = sum(v.nbytes for v in cache.values()) / 1024 ** 2
    print(f"Cache ready: {len(cache)} spectrograms, ~{total_mb:.0f} MB")
    return cache


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_audio(filepath):
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    target = SAMPLE_RATE * DURATION
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio


def audio_to_melspectrogram(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=F_MAX)
    return librosa.power_to_db(mel, ref=np.max)


def encode_labels(primary_label, secondary_labels_str, label_to_idx, num_classes):
    vec = np.zeros(num_classes, dtype=np.float32)
    if primary_label in label_to_idx:
        vec[label_to_idx[primary_label]] = 1.0
    try:
        for sec in ast.literal_eval(secondary_labels_str):
            if sec in label_to_idx:
                vec[label_to_idx[sec]] = 1.0
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
                mel = self.cache[row["filepath"]].copy()
                if self.augment:
                    mel = self._spec_augment(mel)
            else:
                audio = load_audio(row["filepath"])
                if self.augment:
                    audio = self._augment(audio)
                mel = audio_to_melspectrogram(audio)
                if self.augment:
                    mel = self._spec_augment(mel)
            X.append(mel)
            y.append(encode_labels(
                row["primary_label"], row["secondary_labels"],
                self.label_to_idx, self.num_classes
            ))
        return np.array(X)[..., np.newaxis], np.array(y)

    def _augment(self, audio):
        # Gaussian noise — simulates field recording conditions
        audio = audio + np.random.randn(len(audio)) * np.random.uniform(0.001, 0.015)
        # Random gain
        audio = audio * np.random.uniform(0.7, 1.3)
        # Random time shift up to ±0.5 s
        shift = np.random.randint(-SAMPLE_RATE // 2, SAMPLE_RATE // 2)
        audio = np.roll(audio, shift)
        return audio.astype(np.float32)

    def _spec_augment(self, mel):
        mel = mel.copy()
        n_mels, n_time = mel.shape
        fill = mel.min()
        # Frequency masking — 2 masks up to 10 bins each
        for _ in range(2):
            w = np.random.randint(1, 10)
            f0 = np.random.randint(0, max(1, n_mels - w))
            mel[f0:f0 + w, :] = fill
        # Time masking — 2 masks up to 40 steps each
        for _ in range(2):
            w = np.random.randint(1, 40)
            t0 = np.random.randint(0, max(1, n_time - w))
            mel[:, t0:t0 + w] = fill
        return mel

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
            taxon_id = taxon_id.strip()
            try:
                key = int(taxon_id)  # label_to_idx keys are integers from taxonomy CSV
            except ValueError:
                key = taxon_id
            if key in label_to_idx:
                vec[label_to_idx[key]] = 1.0

        filepath = soundscape_dir + row["filename"]
        if os.path.exists(filepath):
            rows.append({"filepath": filepath, "start_sec": start_sec, "label_vector": vec})

    result = pd.DataFrame(rows)
    matched = sum(r["label_vector"].sum() > 0 for _, r in result.iterrows())
    print(f"Soundscape windows: {len(result)} | with at least one mapped species: {matched}")
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
                X.append(self.cache[key].copy())
            else:
                audio, _ = librosa.load(
                    row["filepath"], sr=SAMPLE_RATE,
                    duration=DURATION, offset=row["start_sec"]
                )
                target = SAMPLE_RATE * DURATION
                if len(audio) < target:
                    audio = np.pad(audio, (0, target - len(audio)))
                X.append(audio_to_melspectrogram(audio))
            y.append(row["label_vector"])
        return np.array(X)[..., np.newaxis], np.array(y)

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
    """Stops training after `patience` epochs without val_auc improving by
    `min_rel_delta` *relative* to the current best (e.g. 0.01 = 1% relative gain).
    Relative thresholds adapt as AUC rises: a 0.01 jump at 0.50 is easy, at 0.90 is not.
    """

    def __init__(self, patience=4, min_rel_delta=0.01):
        super().__init__()
        self.patience        = patience
        self.min_rel_delta   = min_rel_delta
        self.best_auc        = 0.0
        self.wait            = 0
        self.plateau_reached = False
        self.epochs_trained  = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_trained = epoch + 1
        current = (logs or {}).get("val_auc", 0.0)
        # relative improvement; first real value (best_auc==0) always counts
        required = self.best_auc * self.min_rel_delta if self.best_auc > 0 else 0.0
        if current - self.best_auc > required:
            self.best_auc = current
            self.wait = 0
        else:
            self.wait += 1
            print(f"    [plateau] val_auc stable for {self.wait}/{self.patience} epochs")
            if self.wait >= self.patience:
                self.plateau_reached = True
                self.model.stop_training = True
                print(f"    [plateau] val_auc plateaued — stopping after {self.epochs_trained} epochs")


# ── Backbone ───────────────────────────────────────────────────────────────────

def build_backbone(name, input_shape=None):
    if input_shape is None:
        input_shape = (N_MELS, 313, 1)  # freq dim must track N_MELS
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Concatenate(axis=-1)([inputs, inputs, inputs])

    if name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", pooling="avg"
        )
        feature_dim = 1280
    elif name == "mobilenet":
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", pooling="avg"
        )
        feature_dim = 1280
    elif name == "resnet":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", pooling="avg"
        )
        feature_dim = 2048
    elif name == "scratch":
        return None, None
    else:
        raise ValueError(f"Unknown backbone: {name}")

    backbone_model = tf.keras.Model(inputs=inputs, outputs=base(x), name=f"backbone_{name}")
    base.trainable = FINE_TUNE
    print(f"Backbone: {BACKBONE_CLASS_NAME[name]} | feature_dim={feature_dim} | trainable={FINE_TUNE}")
    return backbone_model, feature_dim

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
    return match.group(1).strip() if match else None


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


def ask_llm_to_fix(code, error, attempt):
    return call_llm(f"""The Keras model head you generated crashed with this error:

ERROR:
{error}

FAILED CODE:
```python
{code}
```

Fix attempt {attempt}/{MAX_FIX_RETRIES}. Rules reminder:
- Use `backbone_model` as-is — do not define any classes
- Available layers: Dense, Dropout, BatchNormalization, LayerNormalization, Activation,
  Conv1D, Conv2D, DepthwiseConv2D, SeparableConv2D,
  MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, GlobalAveragePooling2D,
  GlobalMaxPooling1D, GlobalMaxPooling2D, Flatten, Reshape, Lambda,
  Concatenate, Add, Multiply, MultiHeadAttention, Attention, Input, Model
- No model.compile() or model.fit()
- Final variable must be named `model`

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

        default_lr = 1e-4 if FINE_TUNE else 1e-3
        lr   = ns.get("learning_rate", default_lr)
        loss = ns.get("loss", "binary_crossentropy")

        # LLM can unfreeze top N layers of backbone by setting fine_tune_layers = N
        fine_tune_n = ns.get("fine_tune_layers", 0)
        if fine_tune_n and fine_tune_n > 0:
            backbone_model.trainable = True
            for layer in backbone_model.layers[:-fine_tune_n]:
                layer.trainable = False
            print(f"  Fine-tuning top {fine_tune_n} backbone layers")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
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
        plateau_cb    = PlateauCallback(patience=5, min_rel_delta=0.005)
        best_wts_cb   = _BestWeightsCallback()
        history = model.fit(
            train_generator, validation_data=val_generator,
            epochs=N_EPOCHS, verbose=1, callbacks=[plateau_cb, best_wts_cb]
        )

        # best (auc, weights) seen so far across all phases
        best_auc     = best_wts_cb.best_auc
        best_weights = best_wts_cb.best_weights

        # ── Phase 2 (only when SOUNDSCAPE_VAL=False): plateau-gated soundscapes ──
        remaining = N_EPOCHS - plateau_cb.epochs_trained
        if plateau_cb.plateau_reached and soundscape_generator is not None and remaining > 0:
            print(f"  Phase 2: {remaining} epoch(s) on train_audio + soundscape combined...")
            combined_generator = CombinedGenerator(train_generator, soundscape_generator)
            plateau_cb2  = PlateauCallback(patience=3, min_rel_delta=0.01)
            best_wts_cb2 = _BestWeightsCallback()
            h2 = model.fit(
                combined_generator, validation_data=val_generator,
                epochs=remaining, verbose=1, callbacks=[plateau_cb2, best_wts_cb2]
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
    print(f"LLM={LLM_MODEL} | Backbone={BACKBONE_CLASS_NAME[BACKBONE]} | Iterations={N_ITERATIONS}")
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
        _fdim = int(backbone_model.output_shape[-1]) if backbone_model is not None else feature_dim
        prompt = build_prompt(_fdim, backbone_name=BACKBONE_CLASS_NAME[BACKBONE], history=history, n_mels=N_MELS)

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
                    label=f"{BACKBONE}_iter{iteration}", architecture="", code="",
                    status="crashed", crash_count=0, error=f"LLM unavailable: {e}",
                    val_auc=None, val_loss=None, epochs_trained=0,
                    training_time_sec=None, llm_analysis="", epoch_history={},
                    notes=f"backbone={BACKBONE}, fine_tune={FINE_TUNE}, epochs={N_EPOCHS}",
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
                fix_response = ask_llm_to_fix(code, result["error"], crash_count)
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
            "label"        : f"{BACKBONE}_iter{iteration}",
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
                notes             = f"backbone={BACKBONE}, fine_tune={FINE_TUNE}, epochs={N_EPOCHS}",
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
    _check_versions()
    df = load_data()
    num_classes, label_to_idx = load_taxonomy()
    train_df, val_df = make_splits(df)

    backbone_model, feature_dim = build_backbone(BACKBONE)

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

    spec_cache = {}
    if not DEBUG:
        all_df = pd.concat([train_df, val_df], ignore_index=True)
        spec_cache = precompute_spectrograms(all_df, soundscape_df)

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
        train_df, label_to_idx, num_classes, augment=True, shuffle=True, cache=spec_cache
    )

    if use_ss_val:
        # Split soundscapes into train/val; train on clips + soundscape-train
        # from epoch 1, validate on the held-out soundscape-val (the real
        # objective). Phase 2 is disabled (soundscape_generator=None) because
        # soundscapes are already mixed into training throughout.
        ss_train_df, ss_val_df = train_test_split(
            soundscape_df, test_size=SOUNDSCAPE_VAL_FRAC, random_state=42
        )
        ss_train_gen    = SoundscapeGenerator(ss_train_df, shuffle=True,  cache=spec_cache)
        train_generator = CombinedGenerator(clip_train_gen, ss_train_gen)
        val_generator   = SoundscapeGenerator(ss_val_df, shuffle=False, cache=spec_cache)
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
            val_df, label_to_idx, num_classes, augment=False, shuffle=False, cache=spec_cache
        )
        soundscape_generator = None
        if soundscape_df is not None and len(soundscape_df) >= BATCH_SIZE:
            soundscape_generator = SoundscapeGenerator(soundscape_df, shuffle=True, cache=spec_cache)
            print(f"Soundscape batches/epoch: {len(soundscape_generator)}")
        print(f"Train samples: {len(train_df)} | batches/epoch: {len(train_generator)}")
        print(f"Val   samples: {len(val_df)}   | batches/epoch: {len(val_generator)}")

    X_batch, y_batch = train_generator[0]
    print(f"Sanity check — X: {X_batch.shape} | y: {y_batch.shape}")

    run_agent(backbone_model, feature_dim, train_generator, val_generator, soundscape_generator)
