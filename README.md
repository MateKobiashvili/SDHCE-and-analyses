# SDHCE — Symbolic Distillation via Hierarchical Concept Extraction

Extracts a human-readable concept hierarchy directly from trained neural network weights. No data required after training.

---

## Usage

```
python sdhce.py dataset.csv hyperparams.txt output.txt
python sdhce.py dataset.csv hyperparams.txt output.txt --autoname
python sdhce.py dataset.csv hyperparams.txt output.txt --autoname --ollama-model llama3.2:3b
python sdhce.py dataset.csv hyperparams.txt output.txt --autoname --ollama-url http://192.168.1.5:11434/api/chat
```

---

## hyperparams.txt

```
input_dim      = 4
output_dim     = 3
hidden_dims    = 8,4
# comma-separated hidden layer sizes
activation     = silu
# silu | relu | tanh | sigmoid
epochs         = 2000
lr             = 0.01
tau_percentile = 0
# 0 = keep all weights | 60 = keep strongest 40%
task           = classification
# classification | regression | multilabel
target_col     = last
# column index or "last" (classification/regression)
# target_cols  = last:3
# multilabel: last N columns
# target_cols  = 4,5,6
# multilabel: explicit column indices
```

Lines starting with `#` are ignored. Order does not matter.

---

## Activation recommendations

| Activation | Best for |
|---|---|
| `silu` | Default. Handles negative inputs, converges fast. |
| `relu` | Avoid on standardized data — dead neurons from negative inputs. |
| `tanh` | Slower convergence, needs Xavier init, higher epochs. |
| `sigmoid` | Binary output layers only. |

---

## Architecture sizing

Keep total parameters within 1–10x your sample count to avoid overfitting.

| Samples | Recommended hidden_dims |
|---|---|
| ~150 (Iris) | 8,4 or 4,2 |
| ~440 (Diabetes) | 32,16 or 16,8 |
| ~1000 | 64,32 or 32,16,8 |

---

## Output: naming conventions

The hierarchy has three node types:

**ATOMS (Level 1)** — depend only on raw input features.
Name by the combination that triggers them, not what they output.

**CONCEPTS (Level 2+)** — depend on atoms or other concepts.
Name by which upstream concepts fire or are suppressed.

**OUTPUTS** — the final prediction nodes. Named by class label automatically.

### Polarity

- `+` dep = that input firing **helps** activate this node
- `-` dep = that input firing **suppresses** this node

### Threshold

- Large positive threshold = fires rarely (extreme/rare detector)
- Near-zero threshold = fires easily (broad sensor)
- Negative threshold = fires by default unless suppressed

### Naming tips

- Lead with the dominant `+` dep, then note dominant `-` dep
- `+ petal_length, - sepal_width` → `long_petal_narrow_sepal`
- All `-` deps → node fires when everything is absent/small → `small_X` or `low_X`
- Mixed signs with similar weights → relative comparison → `petal_over_sepal`
- High threshold + all `-` deps → rare suppression pattern → `extreme_low_X`
- Keep names to 2–4 words in snake_case

---

## Autoname (--autoname)
#NOT RECCOMENDED with <= 3B model size, prompt may need adjusting depending on model
Requires [Ollama](https://ollama.com/download) running locally.

```
ollama pull llama3.2:3b    # fast, ~2GB, good enough
ollama pull phi4           # slower, ~9GB, better names
```


Names are generated one neuron at a time in level order — so by the time a concept neuron is named, its atom dependencies already have real names and are passed as context. Duplicate names are automatically suffixed (`_2`, `_3`, etc.).

If Ollama is not running, the script falls back to placeholder names (`C1_0`, `C1_1`, ...) silently.

---

## Validation

After extraction, SDHCE re-evaluates the symbolic hierarchy on the training data and compares it to the original network.

- **Agreement 100%** — the hierarchy is a perfect symbolic replica of the network.
- **Agreement < 100%** — floating point or activation approximation error. Usually < 0.01% disagreement.
- **RMSE disagreement = 0.0000** for regression — perfect.

The symbolic hierarchy can be transcribed into a plain handcrafted program. The network can then be deleted; the knowledge survives in the program.
