# SDHCE — Symbolic Distillation via Hierarchical Concept Extraction

Extracts a human-readable concept hierarchy directly from trained neural network weights. No data required after training.

### NEW VERSION!
# SDHCE CAN NOW DO THE FOLLOWING:
# 1) AUTONAMING WITH "CONCEPT ARITHMETIC" — terms cancel across layers
#    (e.g. high_petal_length firing into a negative dep becomes low_petal_length,
#     and opposing signals for the same feature collapse to nothing)
# 2) ARBITRARY INTERVAL SCALING — n_intervals=5 auto-generates
#    low / mid_low1 / mid / mid_high1 / high with no manual naming needed
# 3) SYMBOLIC DISTILLATION CHECK — verifies that the named concept layer
#    alone (inputs -> concepts -> output, skipping all intermediate layers)
#    fully reproduces the network's predictions
# 4) SYMBOLIC FORMULA OUTPUT — at the end of output.txt, a self-contained
#    mathematical formula is printed: each concept expanded to raw input
#    weights, and the output layer expressed over those concepts.
#    You can implement it by hand and delete the network.

---
# LIST OF ANALYZED DATASETS/MODELS
```
    1) IRIS CLASSIFICATION
    MODEL TYPE:  4 -> 4 -> 2 -> 3
    ACTIVATION TYPE: SILU
    FILES:
    iris.csv                    # DATASET
    iris_hyperparams.txt        # HYPERPARAMETERS FILE
    iris_output_analyzed.txt    # HUMAN-BASED ANALYSIS DONE ON THE OUTPUT
    ------
    TO GENERATE AN OUTPUT.TXT FILE, YOU SHOULD RUN THIS:
        python sdhce11.py iris.csv iris_hyperparams.txt iris_output.txt --autoname
```

---

## Usage

```
python sdhce11.py dataset.csv hyperparams.txt output.txt
python sdhce11.py dataset.csv hyperparams.txt output.txt --autoname
python sdhce11.py iris.csv iris_hyperparams.txt iris_output.txt --autoname
```

`--autoname` enables symbolic interval naming (no LLM required). Without it,
neurons are labelled with placeholders (`C1_0`, `C1_1`, ...) for manual naming.

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
n_intervals    = 3
# number of bins per feature for autoname (default 3)
# auto-generates labels: 3 -> low/mid/high, 5 -> low/mid_low1/mid/mid_high1/high
# override with: interval_labels = very_low,low,mid,high,very_high
max_deps_in_name = 3
# how many top dependencies to include in each neuron's name (default 3)
```

Lines starting with `#` are ignored. Order does not matter.

---

## Interval labels

When `--autoname` is used, each input feature is divided into `n_intervals` bins
based on training data percentiles. The bin a feature falls into determines the
label prefix (`low_`, `mid_`, `high_`, etc.) used in neuron names.

| n_intervals | Auto-generated labels |
|---|---|
| 2 | low, high |
| 3 | low, mid, high |
| 4 | low, mid_low1, mid_high1, high |
| 5 | low, mid_low1, mid, mid_high1, high |
| 6 | low, mid_low2, mid_low1, mid_high1, mid_high2, high |

Labels are always symmetric and anchored at `low` / `high`. You can override
them with `interval_labels = ...` in the hyperparams file.

### Concept arithmetic and term cancellation

When naming deeper neurons, SDHCE expands each dependency all the way back to
raw input features and sums signed contributions. If the same feature is pulled
in opposite directions by different paths, the signals cancel and that feature
is dropped from the name entirely. This means a Level 2 concept name reflects
the *net* effect on each input — not a concatenation of its deps' names.

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
With `--autoname`, names are derived by expanding all paths back to raw inputs
and cancelling opposing signals before labelling.

**OUTPUTS** — the final prediction nodes. Named by class label automatically.

### Polarity

- `+` dep = that input firing **helps** activate this node
- `-` dep = that input firing **suppresses** this node

### Threshold

- Large positive threshold = fires rarely (extreme/rare detector)
- Near-zero threshold = fires easily (broad sensor)
- Negative threshold = fires by default unless suppressed

### Naming tips (manual mode)

- Lead with the dominant `+` dep, then note dominant `-` dep
- `+ petal_length, - sepal_width` → `long_petal_narrow_sepal`
- All `-` deps → node fires when everything is absent/small → `small_X` or `low_X`
- Mixed signs with similar weights → relative comparison → `petal_over_sepal`
- High threshold + all `-` deps → rare suppression pattern → `extreme_low_X`
- Keep names to 2–4 words in snake_case

---

## Validation and distillation check

After extraction, SDHCE runs two checks:

**Validation** — re-evaluates the full symbolic hierarchy on training data and
compares to the original network.
- **Agreement 100%** — the hierarchy is a perfect symbolic replica.
- **Agreement < 100%** — usually floating point noise, < 0.01% disagreement.

**Distillation check** — evaluates *only* the deepest named concept layer:
inputs → concepts → output, skipping all intermediate layers.
- **SUCCESS** — the named concepts alone fully reproduce the network.
  The intermediate layers were just a computational path; the concepts are the
  complete decision logic.
- **INCOMPLETE** — intermediate layers carry information not captured in the
  concept names. Consider increasing `n_intervals`, lowering `tau_percentile`,
  or adding more hidden layers so concepts form more cleanly.

---

## Symbolic formula

At the end of `output.txt`, a self-contained formula is printed:

```
CONCEPTS (inputs -> concept activations):

  high_petal_length__low_sepal_width__thr+0.76
    = SILU( (+1.2341)*[petal length] + (-0.8821)*[sepal width] + ... + (+0.756) )

OUTPUT (argmax over classes):

  score[0] = (+3.14)*[concept_0] + (-2.40)*[concept_1] + (+7.457)
  score[1] = ...
  prediction = argmax( score[0], score[1], score[2] )
```

Each concept is fully expanded to raw input weights. The output layer shows the
concept-to-class mapping. The whole formula can be implemented by hand —
the trained network can then be deleted. The knowledge survives in the formula.
