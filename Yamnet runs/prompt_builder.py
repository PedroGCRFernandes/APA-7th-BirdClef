def build_prompt(feature_dim=1024, backbone_name="YAMNet", n_frames=9, history=None):
    """
    history: list of dicts from experiment_log, most recent last.
             Each dict has: label, val_auc, val_loss, status, error, architecture, notes
    n_frames: number of YAMNet output frames per 5-s clip (probed at runtime).
    """

    history_section = _format_history(history)
    input_shape = f"({n_frames}, {feature_dim})"

    return f"""You are an ML researcher designing model heads for a bird species audio classifier.

TASK
----
Write a Keras model head for multi-label classification of 234 bird species.

A frozen {backbone_name} audio model has already encoded each 5-second clip into a
SEQUENCE of frame-level embeddings. `backbone_model` is an identity passthrough
over those frames — it has no trainable weights, but the TIME AXIS is yours to
work with.
- Input shape : {input_shape}  — {n_frames} time frames × {feature_dim}-d {backbone_name} embedding
- Output shape: {input_shape}  — same sequence, unchanged

You have full freedom in how you collapse, transform, or combine the frame
embeddings before classifying. The 1024-d {backbone_name} embedding is "raw material"
— apply normalisation, learnable projections, channel-wise gating, parallel
processing branches, temporal attention, learnable pooling, anything that makes
sense for fine-grained bird species ID.

MANDATORY — your code MUST start with exactly these two lines (do not modify):

    inputs = tf.keras.Input(shape={input_shape})
    x      = backbone_model(inputs, training=False)

MANDATORY — your code MUST end by building `model` from `inputs` and a final
`Dense(234, activation='sigmoid')` output. `backbone_model` is a Model object:
NEVER pass it as a positional argument to a layer call or to `tf.keras.Model(...)`.
Only tensors (like `x`, `inputs`, `outputs`) may be passed positionally to layers.

MANDATORY — shape rules for merges:
- `Add()([a, b])` requires `a` and `b` to have IDENTICAL shapes. If you project
  to a different width with `Dense(N)`, you must project BOTH branches to N
  (or skip the residual). The 1024-d YAMNet output cannot be Added to a Dense(512)
  branch — that crashes with "Cannot broadcast 1024 to 512".
- `Concatenate()([a, b])` requires matching shapes on every axis EXCEPT the
  concat axis (default last).
- `Multiply()([a, b])` requires shapes that broadcast — usually identical.

    # your head design — collapse time, transform features, classify
    # e.g. x = GlobalAveragePooling1D()(x)             # simple temporal pool
    # or:  x = MultiHeadAttention(4, 64)(x, x); x = GlobalAveragePooling1D()(x)
    # or:  x = Conv1D(256, 3, padding='same', activation='gelu')(x); x = GlobalMaxPooling1D()(x)
    outputs = Dense(234, activation='sigmoid')(x)
    model   = Model(inputs, outputs)

AVAILABLE VARIABLES
-------------------
Your code runs in a namespace with exactly these variables pre-loaded:
- `backbone_model` — the pretrained {backbone_name} feature extractor described above
- `tf`             — TensorFlow / Keras
- `np`             — NumPy
- Layer shortcuts (use these directly without tf.keras.layers prefix):
  Dense, Dropout, BatchNormalization, LayerNormalization, Activation
  Conv1D, Conv2D, DepthwiseConv2D, SeparableConv2D
  MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D
  GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D
  Flatten, Reshape, Lambda, Concatenate, Add, Multiply
  MultiHeadAttention, Attention
  Input, Model

The agent will call model.compile() and model.fit(train_generator, val_generator) automatically
after your code runs. Do NOT do this yourself.

OPTIONAL OVERRIDES — set any of these variables in your code to change training behaviour:
- `learning_rate = <float>`   — override the default lr (1e-3).
- `loss = <keras loss>`       — override the default loss. Example for focal loss (helps with rare species / class imbalance):
      loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

TECHNIQUES WORTH TRYING (pick one that hasn't been tried yet per the history below)
-------------------------------------------------------------------------------------
1. Focal loss — replaces BCE; down-weights easy examples, focuses on hard/rare ones:
      loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
2. Squeeze-and-Excitation block — gating that reweights embedding dimensions:
      se  = Dense(feature_dim // 16, activation='relu')(x)
      se  = Dense(feature_dim, activation='sigmoid')(se)
      x   = Multiply()([x, se])
3. Vary the hidden activation — past runs only ever used 'relu' (or linear).
   Deliberately try a different one and state which in your comment:
      x = Dense(512, activation='gelu')(x)      # also: 'elu', 'swish', 'selu'
      # or:  x = Dense(512)(x); x = tf.keras.layers.LeakyReLU(0.1)(x)
   (The output layer must stay Dense(234, activation='sigmoid') — multi-label.)

RULES
-----
- Use `backbone_model` as-is — do not redefine, subclass, or replace it
- Do not call model.compile() or model.fit() — the agent handles training
- Do not reference train_generator or val_generator — they are not in your namespace
- The final variable must be named `model`
- No custom class definitions
{history_section}
Return only the Python code in a ```python``` block. No explanations.
"""


def _detect_stagnation(successful, window=3, eps=5e-4):
    """Stagnant if (a) we have ≥ `window`+1 successful runs AND (b) the global
    best val_auc has not improved by more than `eps` in the last `window` runs.

    Returns (stagnant: bool, best_auc: float|None, since_best: int).
    `since_best` = how many runs ago (in iteration order) the best was found.
    """
    if len(successful) < window + 1:
        return False, None, 0

    ordered = sorted(successful, key=lambda e: e.get("timestamp") or e["label"])
    aucs = [e["val_auc"] for e in ordered]
    best_so_far = max(aucs)
    best_idx    = max(i for i, a in enumerate(aucs) if a == best_so_far)
    since_best  = len(aucs) - 1 - best_idx

    recent_max = max(aucs[-window:])
    stagnant   = since_best >= window and (best_so_far - recent_max) >= -eps
    return stagnant, best_so_far, since_best


def _format_history(history):
    if not history:
        return "\nThis is the first experiment — propose a solid baseline head.\n"

    successful = [e for e in history if e["status"] == "success" and e["val_auc"] is not None]
    failed     = [e for e in history if e["status"] == "crashed"]

    lines = ["\nPAST EXPERIMENTS\n----------------"]

    if successful:
        lines.append("Completed runs (best first):")
        for e in sorted(successful, key=lambda x: x["val_auc"], reverse=True):
            code_snippet = (e.get("code") or "")[:200].replace("\n", " | ")
            lines.append(
                f"  - {e['label']}: val_auc={e['val_auc']:.4f} | val_loss={e['val_loss']:.4f}"
                f"\n    code: {code_snippet}"
            )

    if failed:
        lines.append("Failed runs (do not repeat these):")
        for e in failed:
            lines.append(f"  - {e['label']}: {e['error'][:120]}")

    if successful:
        best = max(successful, key=lambda x: x["val_auc"])
        lines.append(
            f"\nBest so far: {best['label']} with val_auc={best['val_auc']:.4f}."
        )

    stagnant, best_auc, since_best = _detect_stagnation(successful)
    if stagnant:
        lines.append(
            f"\n⚠ PLATEAU DETECTED — best val_auc ({best_auc:.4f}) has not improved in the"
            f" last {since_best} successful runs. Small variations on the existing recipe"
            " are not paying off. Make a STRUCTURAL change this iteration. Pick ONE of:\n"
            "  (a) Different loss — asymmetric loss for multi-label, or label-smoothed BCE:\n"
            "        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)\n"
            "  (b) Multi-branch head — process the embedding through 2-3 parallel paths\n"
            "      with different widths/activations, then Concatenate before the output.\n"
            "  (c) Self-attention over the embedding — Reshape (1024,) → (16, 64) (or\n"
            "      (32, 32)), apply MultiHeadAttention, Flatten, then dense output.\n"
            "  (d) Very different learning rate — past runs cluster near 1e-3. Try 3e-4\n"
            "      with a deeper head, or 3e-3 with stronger Dropout(0.5+).\n"
            "  (e) Heavier regularisation — Dropout(0.5), kernel_regularizer=l2(1e-4)\n"
            "      on every Dense, plus LayerNormalization between blocks.\n"
            "  (f) Wider+shallower OR narrower+deeper than anything in the history —\n"
            "      whichever direction is least represented above.\n"
            "State which lever (a-f) you chose in the comment, and WHY based on the failure\n"
            "mode of the recent stagnant runs (e.g. overfit, underfit, val_loss diverging)."
        )
    else:
        lines.append(
            "\nPropose a NOVEL architecture not yet tried above — the code snippets show exactly what has"
            " been used. Do not reuse the same layer types, sizes, or structural pattern from any previous run."
            " Explain your reasoning in a comment at the top of the code block."
        )

    return "\n".join(lines) + "\n"
