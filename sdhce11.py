import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import csv

# ── USAGE ─────────────────────────────────────────────────────────────────────
# python sdhce.py dataset.csv hyperparams.txt output.txt [--autoname]
#
# hyperparams.txt format (one per line, order doesn't matter):
#   input_dim = 4
#   output_dim = 3
#   hidden_dims = 8,4
#   activation = silu        # silu | relu | tanh | sigmoid
#   epochs = 2000
#   lr = 0.01
#   tau_percentile = 0       # 0 = keep all weights, 60 = keep top 40%
#   target_col = last        # column index of label, or "last"
#   task = classification    # classification | regression | multilabel
#   target_cols = last       # for multilabel: comma-separated indices or "last:N" for last N cols
#   n_intervals = 3          # how many bins per feature (default 3)
#   interval_labels = low,mid,high  # comma-separated labels, must match n_intervals
#   max_deps_in_name = 3     # top-K deps to include in symbolic name (default 3)
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. PARSE ARGS ─────────────────────────────────────────────────────────────

if len(sys.argv) != 4 or 5 or 6:
    print("Usage: python sdhce.py dataset.csv hyperparams.txt output.txt")


dataset_path, hyperparam_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
extra_args  = sys.argv[4:]
AUTONAME    = "--autoname" in extra_args


# ── 1. LOAD HYPERPARAMS ───────────────────────────────────────────────────────

def parse_hyperparams(path):
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            params[key.strip()] = val.strip()
    return params

hp = parse_hyperparams(hyperparam_path)

INPUT_DIM      = int(hp["input_dim"])
OUTPUT_DIM     = int(hp["output_dim"])
HIDDEN_DIMS    = [int(x) for x in hp["hidden_dims"].split(",")]
ACTIVATION     = hp.get("activation", "silu").lower()
EPOCHS         = int(hp.get("epochs", "2000"))
LR             = float(hp.get("lr", "0.01"))
TAU_PERCENTILE = float(hp.get("tau_percentile", "0"))
TARGET_COL     = hp.get("target_col", "last")
TARGET_COLS    = hp.get("target_cols", None)
TASK           = hp.get("task", "classification").lower()
N_INTERVALS    = int(hp.get("n_intervals", "3"))
MAX_DEPS_NAME  = int(hp.get("max_deps_in_name", "3"))

def auto_interval_labels(n):
    """
    Generate symmetric interval labels for any N.
    Always anchored: first=low, last=high.
    Middle slots fill symmetrically; odd count gets a centre "mid".
      N=2 -> low, high
      N=3 -> low, mid, high
      N=4 -> low, mid_low1, mid_high1, high
      N=5 -> low, mid_low1, mid, mid_high1, high
      N=6 -> low, mid_low2, mid_low1, mid_high1, mid_high2, high
    """
    if n == 1:
        return ["mid"]
    if n == 2:
        return ["low", "high"]
    if n == 3:
        return ["low", "mid", "high"]
    labels    = [""] * n
    labels[0] = "low"
    labels[-1] = "high"
    mid_count  = n - 2
    if mid_count % 2 == 1:
        centre         = mid_count // 2
        labels[1 + centre] = "mid"
        low_slots  = list(range(1, 1 + centre))
        high_slots = list(range(2 + centre, n - 1))
    else:
        low_slots  = list(range(1, 1 + mid_count // 2))
        high_slots = list(range(1 + mid_count // 2, n - 1))
    for rank, idx in enumerate(reversed(low_slots), 1):
        labels[idx] = f"mid_low{rank}"
    for rank, idx in enumerate(high_slots, 1):
        labels[idx] = f"mid_high{rank}"
    return labels

if "interval_labels" in hp:
    INTERVAL_LABELS = [x.strip() for x in hp["interval_labels"].split(",")]
else:
    INTERVAL_LABELS = auto_interval_labels(N_INTERVALS)

assert len(INTERVAL_LABELS) == N_INTERVALS, \
    f"interval_labels length {len(INTERVAL_LABELS)} != n_intervals {N_INTERVALS}"


# ── 2. LOAD DATASET ───────────────────────────────────────────────────────────

with open(dataset_path) as f:
    reader = csv.reader(f)
    rows   = list(reader)

# Detect header
try:
    float(rows[0][0])
    header = None
    data_rows = rows
except ValueError:
    header = rows[0]
    data_rows = rows[1:]

data = np.array([[float(x) for x in r] for r in data_rows if r], dtype=np.float32)

if TASK == "multilabel":
    if TARGET_COLS and TARGET_COLS.startswith("last:"):
        n_targets   = int(TARGET_COLS.split(":")[1])
        target_idxs = list(range(data.shape[1] - n_targets, data.shape[1]))
    elif TARGET_COLS:
        target_idxs = [int(x) for x in TARGET_COLS.split(",")]
    else:
        target_idxs = [data.shape[1] - 1]
    feature_cols = [i for i in range(data.shape[1]) if i not in target_idxs]
    X_raw = data[:, feature_cols]
    y_raw = data[:, target_idxs].astype(np.float32)
    if header:
        INPUT_NAMES = [header[i] for i in feature_cols]
        CLASS_NAMES = [header[i] for i in target_idxs]
    else:
        INPUT_NAMES = [f"x{i}" for i in range(len(feature_cols))]
        CLASS_NAMES = [f"label_{i}" for i in range(len(target_idxs))]
    y = y_raw
else:
    if TARGET_COL == "last":
        target_idx = data.shape[1] - 1
    else:
        target_idx = int(TARGET_COL)
    feature_cols = [i for i in range(data.shape[1]) if i != target_idx]
    X_raw = data[:, feature_cols]
    y_raw = data[:, target_idx]
    if header:
        INPUT_NAMES = [header[i] for i in feature_cols]
    else:
        INPUT_NAMES = [f"x{i}" for i in range(len(feature_cols))]
    if TASK == "classification":
        y_int = y_raw.astype(int)
        CLASS_NAMES = [str(c) for c in sorted(set(y_int))]
        label_map   = {v: i for i, v in enumerate(sorted(set(y_int)))}
        y           = np.array([label_map[v] for v in y_int])
        OUTPUT_DIM = len(CLASS_NAMES)
    else:
        CLASS_NAMES = ["output"]
        y           = y_raw.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw).astype(np.float32)


# ── 3. BUILD AND TRAIN MODEL ──────────────────────────────────────────────────

ACT_MAP = {
    "silu":    nn.SiLU,
    "relu":    nn.ReLU,
    "tanh":    nn.Tanh,
    "sigmoid": nn.Sigmoid,
}
ActClass = ACT_MAP[ACTIVATION]

def build_model(input_dim, hidden_dims, output_dim, act_class):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), act_class()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)

model   = build_model(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, ActClass)
opt     = torch.optim.Adam(model.parameters(), lr=LR)
if TASK == "classification":
    loss_fn = nn.CrossEntropyLoss()
    y_t     = torch.tensor(y)
elif TASK == "regression":
    loss_fn = nn.MSELoss()
    y_t     = torch.tensor(y).unsqueeze(1)
else:  # multilabel
    loss_fn = nn.BCEWithLogitsLoss()
    y_t     = torch.tensor(y)
X_t = torch.tensor(X)

best_loss  = float('inf')
best_state = None

for epoch in range(EPOCHS):
    opt.zero_grad()
    loss = loss_fn(model(X_t), y_t)
    loss.backward()
    opt.step()
    if loss.item() < best_loss:
        best_loss  = loss.item()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if (epoch + 1) % 100 == 0:
        print(f"  epoch {epoch+1}/{EPOCHS}  loss={loss.item():.4f}", flush=True)

model.load_state_dict(best_state)
print(f"  [checkpoint] restored best model (loss={best_loss:.4f})", flush=True)

if TASK == "classification":
    train_metric = (model(X_t).argmax(1) == y_t).float().mean().item()
    train_metric_label = "Training accuracy"
elif TASK == "regression":
    train_metric = nn.MSELoss()(model(X_t), y_t).item() ** 0.5
    train_metric_label = "Training RMSE    "
else:  # multilabel
    preds_ml   = (torch.sigmoid(model(X_t)) > 0.5).float()
    train_metric = (preds_ml == y_t).float().mean().item()
    train_metric_label = "Training accuracy (per-label avg)"


# ── 4. SDHCE EXTRACTION ───────────────────────────────────────────────────────

def extract_concept_graph(model, input_dim, hidden_dims, output_dim,
                          input_names, class_names, tau_percentile):

    layer_specs = []
    linear_idx  = 0
    for i, module in enumerate(model):
        if isinstance(module, nn.Linear):
            name = "output" if linear_idx == len(hidden_dims) else f"hidden{linear_idx+1}"
            layer_specs.append((
                module.weight.detach().numpy(),
                module.bias.detach().numpy(),
                name,
                linear_idx
            ))
            linear_idx += 1

    graph  = {}
    levels = {}

    for i, name in enumerate(input_names):
        nid = f"input:{i}"
        graph[nid]  = {"type": "input", "name_hint": name, "deps": []}
        levels[nid] = 0

    prev_ids = [f"input:{i}" for i in range(input_dim)]

    for W, b, layer_name, layer_idx in layer_specs:
        is_output = (layer_name == "output")
        curr_ids  = []

        for ni, (w_row, bval) in enumerate(zip(W, b)):
            nid = f"{layer_name}:{ni}"
            curr_ids.append(nid)

            tau          = np.percentile(np.abs(w_row), tau_percentile)
            strong_idx   = np.where(np.abs(w_row) >= tau)[0]
            strong_mags  = np.abs(w_row[strong_idx])
            norm_mags    = strong_mags / strong_mags.sum() if strong_mags.sum() > 0 else strong_mags

            deps = []
            for rank, j in enumerate(strong_idx):
                deps.append({
                    "node":     prev_ids[j],
                    "polarity": int(np.sign(w_row[j])),
                    "weight":   float(norm_mags[rank]),
                    "raw_w":    float(w_row[j]),
                })
            deps.sort(key=lambda d: -d["weight"])

            dep_depth = max((levels[d["node"]] for d in deps), default=0)

            graph[nid] = {
                "type":      "output" if is_output else
                             "atom"   if layer_idx == 0 else "concept",
                "layer":     layer_name,
                "neuron":    ni,
                "deps":      deps,
                "bias":      float(bval),
                "threshold": float(-bval),
                "name_hint": class_names[ni] if (is_output and ni < len(class_names)) else
                             f"C{layer_idx+1}_{ni}",
            }
            levels[nid] = dep_depth + 1

        prev_ids = curr_ids

    return graph, levels


graph, levels = extract_concept_graph(
    model, INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM,
    INPUT_NAMES, CLASS_NAMES, TAU_PERCENTILE
)


# ── 4b. SYMBOLIC INTERVAL AUTONAME ───────────────────────────────────────────

def build_interval_map(X_raw, input_names, n_intervals, interval_labels):
    """
    For each input feature, compute percentile bin edges over the raw
    (unscaled) training data and return a lookup:
        { feature_name: (edges, labels) }
    edges has length n_intervals-1 (the internal cut points).
    """
    imap = {}
    percentile_cuts = [100 * i / n_intervals for i in range(1, n_intervals)]
    for i, name in enumerate(input_names):
        col   = X_raw[:, i]
        edges = np.percentile(col, percentile_cuts)
        imap[name] = (edges, interval_labels)
    return imap


def get_feature_contributions(nid, graph):
    """
    Recursively expand a node all the way down to raw input features.
    Returns { feature_name: signed_contribution } where contribution is
    the product of all raw_w values along the path, accumulated additively
    across all paths reaching the same feature.

    Input nodes are base cases: { feature_name: 1.0 }
    """
    node = graph[nid]
    if node["type"] == "input":
        return {node["name_hint"]: 1.0}

    result = {}
    for dep in node["deps"]:
        child_contribs = get_feature_contributions(dep["node"], graph)
        for feat, val in child_contribs.items():
            result[feat] = result.get(feat, 0.0) + dep["raw_w"] * val
    return result


def label_from_net_contribution(net, interval_labels):
    """
    Map a net signed contribution to an interval label.
    Positive → high end, negative → low end, near-zero → mid (cancelled).
    Returns None if the contribution is negligible (cancelled out).
    """
    n        = len(interval_labels)
    mid_idx  = n // 2
    # Threshold below which we consider the term cancelled
    # (relative — if |net| < 5% of scale, drop it)
    CANCEL_EPS = 0.05
    if abs(net) < CANCEL_EPS:
        return None   # cancelled — drop this feature from the name
    if net > 0:
        return interval_labels[-1]       # high
    else:
        return interval_labels[0]        # low


def symbolic_name(nid, node, graph, interval_labels, max_deps):
    """
    Build a deterministic symbolic name for a neuron via term cancellation.

    1. Recursively expand all deps to raw input features with signed contributions.
    2. Aggregate by feature (sum contributions → net signal per feature).
    3. Drop features whose net signal cancels to ~0.
    4. Sort survivors by |net contribution|, take top max_deps.
    5. Map each to an interval label, join with __, append threshold.
    """
    # Step 1+2: aggregate raw feature contributions across all deps
    net_contribs = {}
    for dep in node["deps"]:
        child_contribs = get_feature_contributions(dep["node"], graph)
        for feat, val in child_contribs.items():
            net_contribs[feat] = net_contribs.get(feat, 0.0) + dep["raw_w"] * val

    # Step 3+4: drop cancelled, sort by magnitude, take top K
    survivors = [
        (feat, net)
        for feat, net in net_contribs.items()
        if label_from_net_contribution(net, interval_labels) is not None
    ]
    survivors.sort(key=lambda x: -abs(x[1]))
    top = survivors[:max_deps]

    # Step 5: build name parts
    parts = [
        f"{label_from_net_contribution(net, interval_labels)}_{feat}"
        for feat, net in top
    ]

    base = "__".join(parts) if parts else node["name_hint"]
    thr  = node["threshold"]
    return f"{base}__thr{thr:+.2f}"


def autoname_graph_symbolic(graph, levels, imap, interval_labels, max_deps):
    """Process nodes level-order so deps are always named before dependents."""
    ordered = sorted(
        [nid for nid, n in graph.items() if n["type"] not in ("input", "output")],
        key=lambda nid: levels[nid]
    )
    seen_names = {}
    for nid in ordered:
        node = graph[nid]
        name = symbolic_name(nid, node, graph, interval_labels, max_deps)
        # Deduplicate
        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 1
        graph[nid]["name_hint"] = name


if AUTONAME:
    print("\n  [autoname] computing symbolic interval names...", flush=True)
    imap = build_interval_map(X_raw, INPUT_NAMES, N_INTERVALS, INTERVAL_LABELS)  # kept for future bin-edge use
    autoname_graph_symbolic(graph, levels, imap, INTERVAL_LABELS, MAX_DEPS_NAME)
    print("  [autoname] done\n", flush=True)


# ── 5. VALIDATION ─────────────────────────────────────────────────────────────

ACT_NUMPY = {
    "silu":    lambda z: z / (1.0 + np.exp(-z)),
    "relu":    lambda z: np.maximum(0, z),
    "tanh":    lambda z: np.tanh(z),
    "sigmoid": lambda z: 1 / (1.0 + np.exp(-z)),
}
act_np = ACT_NUMPY[ACTIVATION]

def evaluate_symbolic(graph, levels, X):
    ordered = sorted(
        [nid for nid, n in graph.items() if n["type"] != "input"],
        key=lambda nid: levels[nid]
    )
    preds = []
    for row in X:
        values = {f"input:{i}": float(row[i]) for i in range(len(row))}
        for nid in ordered:
            node = graph[nid]
            z    = sum(dep["raw_w"] * values[dep["node"]] for dep in node["deps"]) + node["bias"]
            values[nid] = z if node["type"] == "output" else act_np(z)
        if TASK == "classification":
            logits = [values[f"output:{i}"] for i in range(len(CLASS_NAMES))]
            preds.append(int(np.argmax(logits)))
        elif TASK == "regression":
            preds.append(float(values["output:0"]))
        else:  # multilabel
            preds.append([float(values[f"output:{i}"]) for i in range(len(CLASS_NAMES))])
    return np.array(preds)

with torch.no_grad():
    if TASK == "classification":
        net_preds = model(X_t).argmax(1).numpy()
    elif TASK == "regression":
        net_preds = model(X_t).squeeze(1).numpy()
    else:  # multilabel
        net_preds = (torch.sigmoid(model(X_t)) > 0.5).float().numpy()

sym_preds = evaluate_symbolic(graph, levels, X)

if TASK == "classification":
    net_acc    = float(np.mean(net_preds == y))
    sym_acc    = float(np.mean(sym_preds == y))
    agreement  = float(np.mean(net_preds == sym_preds))
    mismatches = np.where(net_preds != sym_preds)[0]
elif TASK == "regression":
    net_acc    = float(np.sqrt(np.mean((net_preds - y) ** 2)))
    sym_acc    = float(np.sqrt(np.mean((sym_preds - y) ** 2)))
    agreement  = float(np.sqrt(np.mean((net_preds - sym_preds) ** 2)))
    mismatches = np.array([])
else:  # multilabel
    sym_bin    = (sym_preds > 0.5).astype(float)
    net_acc    = float(np.mean(net_preds == y))
    sym_acc    = float(np.mean(sym_bin == y))
    agreement  = float(np.mean(net_preds == sym_bin))
    mismatches = np.array([])


# ── 5b. DISTILLATION CHECK ────────────────────────────────────────────────────
# Evaluate: inputs -> named concept nodes (deepest hidden level) -> output only.
# If this matches the full network, the named concepts are the complete story.

def evaluate_concept_only(graph, levels, X):
    """
    Two-step evaluation using only the deepest named concept layer:
      1. Compute concept activations recursively from inputs.
      2. Feed directly into output nodes.
    """
    concept_level = max(
        levels[nid]
        for nid, n in graph.items()
        if n["type"] not in ("input", "output")
    )
    concept_nids = sorted(
        [nid for nid, n in graph.items()
         if n["type"] not in ("input", "output") and levels[nid] == concept_level],
        key=lambda nid: graph[nid]["neuron"]
    )
    output_nids = sorted(
        [nid for nid, n in graph.items() if n["type"] == "output"],
        key=lambda nid: graph[nid]["neuron"]
    )

    def eval_node(nid, input_values, cache):
        if nid in cache:
            return cache[nid]
        node = graph[nid]
        if node["type"] == "input":
            return input_values[nid]
        z   = sum(dep["raw_w"] * eval_node(dep["node"], input_values, cache)
                  for dep in node["deps"]) + node["bias"]
        val = z if node["type"] == "output" else act_np(z)
        cache[nid] = val
        return val

    preds = []
    for row in X:
        input_values = {f"input:{i}": float(row[i]) for i in range(len(row))}
        cache        = {}

        # Step 1: activate concept layer
        concept_acts = {nid: eval_node(nid, input_values, cache) for nid in concept_nids}

        # Step 2: output logits directly from concept activations
        output_logits = []
        for out_nid in output_nids:
            out_node = graph[out_nid]
            z = sum(
                dep["raw_w"] * concept_acts[dep["node"]]
                for dep in out_node["deps"]
                if dep["node"] in concept_acts
            ) + out_node["bias"]
            output_logits.append(z)

        if TASK == "classification":
            preds.append(int(np.argmax(output_logits)))
        elif TASK == "regression":
            preds.append(float(output_logits[0]))
        else:
            preds.append([float(v) for v in output_logits])

    return np.array(preds)

concept_preds = evaluate_concept_only(graph, levels, X)

if TASK == "classification":
    concept_acc        = float(np.mean(concept_preds == y))
    concept_agreement  = float(np.mean(concept_preds == net_preds))
    concept_mismatches = np.where(concept_preds != net_preds)[0]
elif TASK == "regression":
    concept_acc        = float(np.sqrt(np.mean((concept_preds - y) ** 2)))
    concept_agreement  = float(np.sqrt(np.mean((concept_preds - net_preds) ** 2)))
    concept_mismatches = np.array([])
else:
    c_bin              = (concept_preds > 0.5).astype(float)
    concept_acc        = float(np.mean(c_bin == y))
    concept_agreement  = float(np.mean(c_bin == net_preds))
    concept_mismatches = np.array([])


# ── 5c. BUILD SYMBOLIC FORMULA ────────────────────────────────────────────────

def build_symbolic_formula(graph, levels, act_name, class_names, task):
    """
    Emit the distilled formula as human-readable nested math.
    Uses the deepest concept layer as the symbolic variables,
    each of which is itself expressed in terms of raw inputs.

    Format:
      concept_name = ACT( w1*feat1 + w2*feat2 + ... + bias )
      ...
      output_class = argmax( w*concept + ... + bias )   [classification]
                   or  ACT( w*concept + ... + bias )    [regression]
    """
    act_str = act_name.upper()

    concept_level = max(
        levels[nid]
        for nid, n in graph.items()
        if n["type"] not in ("input", "output")
    )
    concept_nids = sorted(
        [nid for nid, n in graph.items()
         if n["type"] not in ("input", "output") and levels[nid] == concept_level],
        key=lambda nid: graph[nid]["neuron"]
    )
    output_nids = sorted(
        [nid for nid, n in graph.items() if n["type"] == "output"],
        key=lambda nid: graph[nid]["neuron"]
    )

    lines = []

    # ── concept definitions (inputs -> concept) ──────────────────────────────
    lines.append("  CONCEPTS (inputs -> concept activations):")
    lines.append("")
    for nid in concept_nids:
        node      = graph[nid]
        name      = node["name_hint"]
        # expand to raw input contributions
        net_contribs = {}
        for dep in node["deps"]:
            child = get_feature_contributions(dep["node"], graph)
            for feat, val in child.items():
                net_contribs[feat] = net_contribs.get(feat, 0.0) + dep["raw_w"] * val
        # sort by magnitude
        terms = sorted(net_contribs.items(), key=lambda x: -abs(x[1]))
        term_strs = [f"({w:+.4f})*[{f}]" for f, w in terms]
        bias_str  = f"({node['bias']:+.4f})"
        inner     = " + ".join(term_strs) + " + " + bias_str
        lines.append(f"  {name}")
        lines.append(f"    = {act_str}( {inner} )")
        lines.append("")

    # ── output definitions (concepts -> output) ───────────────────────────────
    if task == "classification":
        lines.append("  OUTPUT (argmax over classes):")
        lines.append("")
        for out_nid in output_nids:
            out_node  = graph[out_nid]
            class_lbl = class_names[out_node["neuron"]] if out_node["neuron"] < len(class_names) else str(out_node["neuron"])
            terms     = []
            for dep in out_node["deps"]:
                dep_name = graph[dep["node"]]["name_hint"]
                terms.append(f"({dep['raw_w']:+.4f})*[{dep_name}]")
            bias_str = f"({out_node['bias']:+.4f})"
            inner    = " + ".join(terms) + " + " + bias_str
            lines.append(f"  score[{class_lbl}] = {inner}")
        lines.append("")
        lines.append("  prediction = argmax( score[" + "], score[".join(
            class_names[graph[nid]["neuron"]] if graph[nid]["neuron"] < len(class_names)
            else str(graph[nid]["neuron"])
            for nid in output_nids
        ) + "] )")
    elif task == "regression":
        out_nid  = output_nids[0]
        out_node = graph[out_nid]
        terms    = []
        for dep in out_node["deps"]:
            dep_name = graph[dep["node"]]["name_hint"]
            terms.append(f"({dep['raw_w']:+.4f})*[{dep_name}]")
        bias_str = f"({out_node['bias']:+.4f})"
        inner    = " + ".join(terms) + " + " + bias_str
        lines.append("  OUTPUT:")
        lines.append("")
        lines.append(f"  prediction = {inner}")
    else:  # multilabel
        lines.append("  OUTPUT (per label, threshold 0.5):")
        lines.append("")
        for out_nid in output_nids:
            out_node  = graph[out_nid]
            class_lbl = class_names[out_node["neuron"]] if out_node["neuron"] < len(class_names) else str(out_node["neuron"])
            terms     = []
            for dep in out_node["deps"]:
                dep_name = graph[dep["node"]]["name_hint"]
                terms.append(f"({dep['raw_w']:+.4f})*[{dep_name}]")
            bias_str = f"({out_node['bias']:+.4f})"
            inner    = " + ".join(terms) + " + " + bias_str
            lines.append(f"  {class_lbl} = sigmoid( {inner} ) > 0.5")

    return lines

symbolic_formula_lines = build_symbolic_formula(
    graph, levels, ACTIVATION, CLASS_NAMES, TASK
)


# ── 6. OUTPUT ─────────────────────────────────────────────────────────────────

def polarity_str(p):
    return "+" if p == 1 else "-"

lines = []
w = lines.append

w("=" * 65)
w("  SDHCE — Symbolic Distillation via Hierarchical Concept Extraction")
w(f"  Dataset   : {dataset_path}")
w(f"  Arch      : {INPUT_DIM} -> {' -> '.join(str(h) for h in HIDDEN_DIMS)} -> {OUTPUT_DIM}")
w(f"  Activation: {ACTIVATION.upper()}   |   tau_percentile: {TAU_PERCENTILE}")
if AUTONAME:
    w(f"  Autoname  : symbolic intervals={N_INTERVALS} labels={','.join(INTERVAL_LABELS)} max_deps={MAX_DEPS_NAME}")
w("=" * 65)
w("")
w(f"  {train_metric_label}: {train_metric*100:.1f}%" if TASK == "classification" else f"  {train_metric_label}: {train_metric:.4f}")
w("")

# Input dims
w("  INPUT DIMENSIONS:")
for i, name in enumerate(INPUT_NAMES):
    w(f"    x{i} = {name}")
w("")

# Concept hierarchy
by_level = defaultdict(list)
for nid, node in graph.items():
    if node["type"] != "input":
        by_level[levels[nid]].append(nid)

for level in sorted(by_level.keys()):
    nodes      = by_level[level]
    level_type = graph[nodes[0]]["type"].upper()
    w("-" * 65)
    w(f"  LEVEL {level}  [{level_type}S]")
    w("-" * 65)
    for nid in nodes:
        node = graph[nid]
        w(f"\n  [{node['name_hint']}]  (threshold = {node['threshold']:+.3f})")
        w("   -> active when:")
        if not node["deps"]:
            w("      (no dependencies above tau)")
        for dep in node["deps"]:
            lbl = graph[dep["node"]].get("name_hint", dep["node"])
            pol = polarity_str(dep["polarity"])
            w(f"      {pol} {lbl:<22} (contributes {dep['weight']*100:.1f}%)")
    w("")

if not AUTONAME:
    w("  NOTE: name_hints are placeholders — a human translator assigns real names.")
    w("")

# Validation
w("=" * 65)
w("  VALIDATION: symbolic hierarchy vs original network")
w("=" * 65)
w("")
if TASK == "classification":
    w(f"  Original network accuracy :  {net_acc*100:.1f}%")
    w(f"  Symbolic hierarchy accuracy: {sym_acc*100:.1f}%")
    w(f"  Agreement with each other  : {agreement*100:.1f}%")
elif TASK == "regression":
    w(f"  Original network RMSE :  {net_acc:.4f}")
    w(f"  Symbolic hierarchy RMSE: {sym_acc:.4f}")
    w(f"  Disagreement (RMSE)   : {agreement:.4f}")
else:  # multilabel
    w(f"  Original network accuracy (per-label) :  {net_acc*100:.1f}%")
    w(f"  Symbolic hierarchy accuracy (per-label): {sym_acc*100:.1f}%")
    w(f"  Agreement with each other (per-label)  : {agreement*100:.1f}%")
w("")

if TASK == "classification":
    if len(mismatches) == 0:
        w("  No mismatches — perfect agreement on every sample.")
    else:
        w(f"  Mismatches ({len(mismatches)} / {len(y)} samples):")
        for i in mismatches[:50]:
            w(f"    sample {i:4d}: network={CLASS_NAMES[net_preds[i]]:<12} symbolic={CLASS_NAMES[sym_preds[i]]}")
        if len(mismatches) > 50:
            w(f"    ... and {len(mismatches)-50} more.")
elif TASK == "regression":
    w("  (Sample-level disagreement shown as RMSE above)")
else:  # multilabel
    w("  (Per-label accuracy shown above)")

w("")

# Distillation check
w("=" * 65)
w("  DISTILLATION CHECK: concept layer -> output (no intermediate layers)")
w("=" * 65)
w("")
if TASK == "classification":
    w(f"  Concept-only accuracy      : {concept_acc*100:.1f}%")
    w(f"  Agreement with full network: {concept_agreement*100:.1f}%")
    if len(concept_mismatches) == 0:
        w("")
        w("  SUCCESS: named concepts fully reproduce the network.")
    else:
        w("")
        w(f"  INCOMPLETE: {len(concept_mismatches)}/{len(y)} samples differ from network.")
        w("    (Intermediate layers carry information not captured in concept names.)")
        for i in concept_mismatches[:20]:
            w(f"    sample {i:4d}: network={CLASS_NAMES[net_preds[i]]:<12} concept={CLASS_NAMES[concept_preds[i]]}")
        if len(concept_mismatches) > 20:
            w(f"    ... and {len(concept_mismatches)-20} more.")
elif TASK == "regression":
    w(f"  Concept-only RMSE          : {concept_acc:.4f}")
    w(f"  Disagreement vs network    : {concept_agreement:.4f}")
    if concept_agreement < 1e-3:
        w("")
        w("  SUCCESS: named concepts fully reproduce the network.")
    else:
        w("")
        w("  INCOMPLETE: concept layer diverges from full network.")
else:  # multilabel
    w(f"  Concept-only accuracy      : {concept_acc*100:.1f}%")
    w(f"  Agreement with full network: {concept_agreement*100:.1f}%")
    if concept_agreement >= 1.0:
        w("")
        w("  SUCCESS: named concepts fully reproduce the network.")
    else:
        w("")
        w("  INCOMPLETE: concept layer diverges from full network.")
w("")
w("=" * 65)

# Symbolic formula
w("=" * 65)
w("  SYMBOLIC FORMULA (distilled)")
w("=" * 65)
w("")
for fl in symbolic_formula_lines:
    w(fl)
w("")
w("=" * 65)

output_text = "\n".join(lines)
print(output_text)

with open(output_path, "w") as f:
    f.write(output_text + "\n")

print(f"\n  [saved to {output_path}]")
