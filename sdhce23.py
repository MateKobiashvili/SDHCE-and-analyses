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
        centre             = mid_count // 2
        labels[1 + centre] = "mid"
        low_slots          = list(range(1, 1 + centre))
        high_slots         = list(range(2 + centre, n - 1))
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

try:
    float(rows[0][0])
    header    = None
    data_rows = rows
except ValueError:
    header    = rows[0]
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
    X_raw        = data[:, feature_cols]
    y_raw        = data[:, target_idx]
    if header:
        INPUT_NAMES = [header[i] for i in feature_cols]
    else:
        INPUT_NAMES = [f"x{i}" for i in range(len(feature_cols))]
    if TASK == "classification":
        y_int       = y_raw.astype(int)
        CLASS_NAMES = [str(c) for c in sorted(set(y_int))]
        label_map   = {v: i for i, v in enumerate(sorted(set(y_int)))}
        y           = np.array([label_map[v] for v in y_int])
        OUTPUT_DIM  = len(CLASS_NAMES)
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
    "linear":  nn.Identity,
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
else:
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
    train_metric       = (model(X_t).argmax(1) == y_t).float().mean().item()
    train_metric_label = "Training accuracy"
elif TASK == "regression":
    train_metric       = nn.MSELoss()(model(X_t), y_t).item() ** 0.5
    train_metric_label = "Training RMSE    "
else:
    preds_ml           = (torch.sigmoid(model(X_t)) > 0.5).float()
    train_metric       = (preds_ml == y_t).float().mean().item()
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
        nid         = f"input:{i}"
        graph[nid]  = {"type": "input", "name_hint": name, "deps": []}
        levels[nid] = 0

    prev_ids = [f"input:{i}" for i in range(input_dim)]

    for W, b, layer_name, layer_idx in layer_specs:
        is_output = (layer_name == "output")
        curr_ids  = []

        for ni, (w_row, bval) in enumerate(zip(W, b)):
            nid          = f"{layer_name}:{ni}"
            curr_ids.append(nid)

            tau         = np.percentile(np.abs(w_row), tau_percentile)
            strong_idx  = np.where(np.abs(w_row) >= tau)[0]
            strong_mags = np.abs(w_row[strong_idx])
            norm_mags   = strong_mags / strong_mags.sum() if strong_mags.sum() > 0 else strong_mags

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
    imap = {}
    percentile_cuts = [100 * i / n_intervals for i in range(1, n_intervals)]
    for i, name in enumerate(input_names):
        col        = X_raw[:, i]
        edges      = np.percentile(col, percentile_cuts)
        imap[name] = (edges, interval_labels)
    return imap


def get_feature_contributions(nid, graph):
    """
    Returns (weights: dict[feat -> float], bias_offset: float)
    bias_offset accumulates all intermediate biases scaled by path weights.
    """
    node = graph[nid]
    if node["type"] == "input":
        return {node["name_hint"]: 1.0}, 0.0

    weights     = {}
    bias_offset = node["bias"]
    for dep in node["deps"]:
        child_w, child_b = get_feature_contributions(dep["node"], graph)
        for feat, val in child_w.items():
            weights[feat] = weights.get(feat, 0.0) + dep["raw_w"] * val
        bias_offset += dep["raw_w"] * child_b
    return weights, bias_offset


def symbolic_name(nid, node, graph, imap, scaler, input_names,
                  interval_labels, max_deps):
    net_contribs = {}
    for dep in node["deps"]:
        child_w, _ = get_feature_contributions(dep["node"], graph)
        for feat, val in child_w.items():
            net_contribs[feat] = net_contribs.get(feat, 0.0) + dep["raw_w"] * val

    total_mag = sum(abs(v) for v in net_contribs.values()) or 1.0

    survivors = []
    for feat, net_w in net_contribs.items():
        if abs(net_w) < 0.05:
            continue
        pct = abs(net_w) / total_mag * 100

        labels = imap[feat][1] if feat in imap else interval_labels
        # positive net_w = neuron fires when feature is HIGH → top of label list
        # negative net_w = neuron fires when feature is LOW  → bottom of label list
        if net_w > 0:
            lbl = labels[-1]
        else:
            lbl = labels[0]

        survivors.append((feat, lbl, abs(net_w), pct))

    survivors.sort(key=lambda x: -x[2])
    top   = survivors[:max_deps]
    parts = [f"{lbl}_{feat}({pct:.0f}%)" for feat, lbl, _, pct in top]
    base  = "__".join(parts) if parts else node["name_hint"]
    return f"{base}__thr{node['threshold']:+.2f}"

def autoname_graph_symbolic(graph, levels, imap, scaler, input_names,
                             interval_labels, max_deps):
    ordered = sorted(
        [nid for nid, n in graph.items() if n["type"] not in ("input", "output")],
        key=lambda nid: levels[nid]
    )
    seen_names = {}
    for nid in ordered:
        node = graph[nid]
        name = symbolic_name(nid, node, graph, imap, scaler, input_names,
                             interval_labels, max_deps)
        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 1
        graph[nid]["name_hint"] = name


if AUTONAME:
    print("\n  [autoname] computing symbolic interval names...", flush=True)
    imap = build_interval_map(X_raw, INPUT_NAMES, N_INTERVALS, INTERVAL_LABELS)
    autoname_graph_symbolic(graph, levels, imap, scaler, INPUT_NAMES,
                            INTERVAL_LABELS, MAX_DEPS_NAME)
    print("  [autoname] done\n", flush=True)


# ── 5. VALIDATION ─────────────────────────────────────────────────────────────

ACT_NUMPY = {
    "silu":    lambda z: z / (1.0 + np.exp(-z)),
    "relu":    lambda z: np.maximum(0, z),
    "tanh":    lambda z: np.tanh(z),
    "sigmoid": lambda z: 1 / (1.0 + np.exp(-z)),
    "linear":  lambda z: z,
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
            node        = graph[nid]
            z           = sum(dep["raw_w"] * values[dep["node"]] for dep in node["deps"]) + node["bias"]
            values[nid] = z if node["type"] == "output" else act_np(z)
        if TASK == "classification":
            logits = [values[f"output:{i}"] for i in range(len(CLASS_NAMES))]
            preds.append(int(np.argmax(logits)))
        elif TASK == "regression":
            preds.append(float(values["output:0"]))
        else:
            preds.append([float(values[f"output:{i}"]) for i in range(len(CLASS_NAMES))])
    return np.array(preds)

with torch.no_grad():
    if TASK == "classification":
        net_preds = model(X_t).argmax(1).numpy()
    elif TASK == "regression":
        net_preds = model(X_t).squeeze(1).numpy()
    else:
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
else:
    sym_bin    = (sym_preds > 0.5).astype(float)
    net_acc    = float(np.mean(net_preds == y))
    sym_acc    = float(np.mean(sym_bin == y))
    agreement  = float(np.mean(net_preds == sym_bin))
    mismatches = np.array([])


# ── 5b. DISTILLATION CHECK ────────────────────────────────────────────────────

def evaluate_concept_only(graph, levels, X):
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
        z          = sum(dep["raw_w"] * eval_node(dep["node"], input_values, cache)
                         for dep in node["deps"]) + node["bias"]
        val        = z if node["type"] == "output" else act_np(z)
        cache[nid] = val
        return val

    preds = []
    for row in X:
        input_values = {f"input:{i}": float(row[i]) for i in range(len(row))}
        cache        = {}
        concept_acts = {nid: eval_node(nid, input_values, cache) for nid in concept_nids}

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

w("  INPUT DIMENSIONS:")
for i, name in enumerate(INPUT_NAMES):
    w(f"    x{i} = {name}")
w("")

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
else:
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
else:
    w("  (Per-label accuracy shown above)")
w("")

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
else:
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

output_text = "\n".join(lines)
print(output_text)

with open(output_path, "w") as f:
    f.write(output_text + "\n")

print(f"\n  [saved to {output_path}]")
