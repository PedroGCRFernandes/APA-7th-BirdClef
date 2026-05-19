def build_prompt(feature_dim=1280, backbone_name="EfficientNetB0", history=None, n_mels=64):
    """
    history: list of dicts from experiment_log, most recent last.
             Each dict has: label, val_auc, val_loss, status, error, architecture, notes
    n_mels: spectrogram frequency bins (agent.py N_MELS) — sets the input shape.
    """

    history_section = _format_history(history)
    input_shape = f"({n_mels}, 313, 1)"

    return f"""You are an ML researcher designing model heads for a bird species audio classifier.

TASK
----
Write a Keras model head for multi-label classification of 234 bird species from audio spectrograms.

A pretrained {backbone_name} backbone is already loaded as `backbone_model`.
- Input shape : {input_shape}  — mel-spectrogram (frequency x time x channel)
- Output shape: ({feature_dim},)  — flat feature vector

A minimal starting pattern — you are free to design the head however you like (branches, skip connections, attention, Conv1D over features, etc.), as long as you call `backbone_model` and assign the final model to `model`:

    inputs  = tf.keras.Input(shape={input_shape})
    x       = backbone_model(inputs, training=False)  # fixed — do not modify backbone_model
    # your head design here
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
- `learning_rate = <float>`   — override the default lr (1e-4). Use a lower lr (e.g. 1e-5) when fine-tuning backbone layers.
- `loss = <keras loss>`       — override the default loss. Example for focal loss (helps with rare species / class imbalance):
      loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
- `fine_tune_layers = <int>`  — unfreeze the top N layers of backbone_model for fine-tuning. Example:
      fine_tune_layers = 20   # unfreeze top 20 layers; use a lower learning_rate too

TECHNIQUES WORTH TRYING (pick one that hasn't been tried yet per the history below)
-------------------------------------------------------------------------------------
1. Focal loss — replaces BCE; down-weights easy examples, focuses on hard/rare ones:
      loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
2. Squeeze-and-Excitation block — channel attention that reweights feature dimensions:
      gap = tf.reduce_mean(x, axis=-1, keepdims=True)  # or use a Dense squeeze/excite pattern
      se  = Dense(feature_dim // 16, activation='relu')(x)
      se  = Dense(feature_dim, activation='sigmoid')(se)
      x   = Multiply()([x, se])
3. Backbone fine-tuning — unfreeze top layers for domain adaptation:
      fine_tune_layers = 20
      learning_rate    = 5e-5

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

    lines.append(
        "\nPropose a NOVEL architecture not yet tried above — the code snippets show exactly what has"
        " been used. Do not reuse the same layer types, sizes, or structural pattern from any previous run."
        " Explain your reasoning in a comment at the top of the code block."
    )

    return "\n".join(lines) + "\n"
