import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import csv
import re

# ensure stdout handles UTF-8 block chars on all platforms (e.g. Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── USAGE ─────────────────────────────────────────────────────────────────────
# python sdhce.py dataset.csv hyperparams.txt output.txt [--autoname] [--ollama-model MODEL] [--ollama-url URL]
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
# ─────────────────────────────────────────────────────────────────────────────


# ── 0. PARSE ARGS ─────────────────────────────────────────────────────────────

if len(sys.argv) < 4:
    print("Usage: python sdhce.py dataset.csv hyperparams.txt output.txt [--autoname] [--ollama-model MODEL] [--ollama-url URL]")
    sys.exit(1)

dataset_path, hyperparam_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
extra_args   = sys.argv[4:]
AUTONAME     = "--autoname" in extra_args
OLLAMA_MODEL = next((extra_args[i+1] for i, a in enumerate(extra_args) if a == "--ollama-model"), "llama3.2")
OLLAMA_URL   = next((extra_args[i+1] for i, a in enumerate(extra_args) if a == "--ollama-url"),   "http://localhost:11434/api/chat")


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
    train_metric = (model(X_t).argmax(1) == y_t).float().mean().item()
    train_metric_label = "Training accuracy"
elif TASK == "regression":
    train_metric = nn.MSELoss()(model(X_t), y_t).item() ** 0.5
    train_metric_label = "Training RMSE    "
else:
    preds_ml   = (torch.sigmoid(model(X_t)) > 0.5).float()
    train_metric = (preds_ml == y_t).float().mean().item()
    train_metric_label = "Training accuracy (per-label avg)"


# ── 4. SDHCE EXTRACTION ───────────────────────────────────────────────────────

def extract_concept_graph(model, input_dim, hidden_dims, output_dim,
                          input_names, class_names, tau_percentile):
    layer_specs = []
    linear_idx  = 0
    for module in model:
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


# ── 4b. DETERMINISTIC ANNOTATION ─────────────────────────────────────────────
#
#  Two complementary signals, both fully deterministic and data-driven:
#
#  1. INPUT CORRELATION
#     For every non-input neuron, collect its post-activation values across all
#     training samples, then compute Pearson r with every raw (scaled) input
#     feature.  The top-k correlated inputs (by |r|) become the annotation.
#     Sign of r tells you direction (positive = neuron fires when feature is
#     high; negative = fires when feature is low).
#
#  2. LAYER CORRELATION
#     Same idea but the "features" are the post-activation values of every
#     neuron in the *immediately preceding layer*.  This tells you which
#     upstream learned concepts this neuron most strongly tracks — not just
#     raw inputs.  Useful for deeper layers where the neuron is composing
#     earlier abstractions.
#
#  Together they give two complementary lenses:
#    - "what raw signal drives this neuron?"  (input correlation)
#    - "what learned sub-concept does it refine?" (layer correlation)
#
#  For classification tasks we also append a class-discriminability tag:
#  the class whose samples produce the highest mean activation for this neuron.

ACT_NUMPY = {
    "silu":    lambda z: z / (1.0 + np.exp(-z)),
    "relu":    lambda z: np.maximum(0, z),
    "tanh":    lambda z: np.tanh(z),
    "sigmoid": lambda z: 1 / (1.0 + np.exp(-z)),
}
act_np = ACT_NUMPY[ACTIVATION]


def collect_layer_activations(model, X, act_np, hidden_dims, input_dim):
    """
    Returns a dict  layer_name -> (N, neurons) ndarray of post-activation values.
    "input" layer is just X itself (already scaled).
    Output layer stores raw logits (pre-softmax / pre-sigmoid).
    """
    activations = {"input": X.copy()}
    prev = X.copy()
    linear_idx = 0

    for module in model:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().numpy()  # (out, in)
            b = module.bias.detach().numpy()    # (out,)
            z = prev @ W.T + b                  # (N, out)
            layer_name = "output" if linear_idx == len(hidden_dims) else f"hidden{linear_idx+1}"
            if layer_name == "output":
                activations[layer_name] = z     # logits, no activation
            else:
                activations[layer_name] = act_np(z)
            prev = activations[layer_name]
            linear_idx += 1

    return activations


def pearson_r_matrix(A, B):
    """
    Compute Pearson r between every column of A and every column of B.
    Returns shape (A.shape[1], B.shape[1]).
    Handles zero-variance columns gracefully (returns 0).
    """
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    std_A = A.std(axis=0)
    std_B = B.std(axis=0)
    # avoid division by zero
    std_A[std_A == 0] = 1e-12
    std_B[std_B == 0] = 1e-12
    A = A / std_A
    B = B / std_B
    return (A.T @ B) / A.shape[0]   # (cols_A, cols_B)


def annotate_graph_deterministic(graph, levels, X, model, act_np,
                                 hidden_dims, input_dim, input_names,
                                 class_names, task, y,
                                 top_k_input=3, top_k_layer=2):
    """
    Adds two annotation fields to every non-input, non-output node:

      node["input_corr"]  — list of dicts  {name, r}  sorted by |r| desc
      node["layer_corr"]  — list of dicts  {name, r}  sorted by |r| desc
      node["class_affinity"] — str  (classification only)

    Also updates name_hint to a compact deterministic label derived from
    the top input correlation, e.g.  "+petal_length"  or  "-sepal_width".
    """
    layer_acts = collect_layer_activations(model, X, act_np, hidden_dims, input_dim)

    # Map layer name -> list of (neuron_index, node_id, node_id_as_name_key)
    # We store the node_id as the "name" in layer_corr so the post-annotation
    # second pass can look up the finalised name_hint later.
    layer_neuron_names = {"input": [(i, f"input:{i}", f"input:{i}") for i in range(len(input_names))]}
    linear_idx = 0
    for module in model:
        if isinstance(module, nn.Linear):
            lname = "output" if linear_idx == len(hidden_dims) else f"hidden{linear_idx+1}"
            out_dim = module.weight.shape[0]
            layer_neuron_names[lname] = [
                (ni, f"{lname}:{ni}", f"{lname}:{ni}")
                for ni in range(out_dim)
            ]
            linear_idx += 1

    # Build ordered list of (layer_name, prev_layer_name)
    layer_order = []
    linear_idx  = 0
    prev_lname  = "input"
    for module in model:
        if isinstance(module, nn.Linear):
            lname = "output" if linear_idx == len(hidden_dims) else f"hidden{linear_idx+1}"
            layer_order.append((lname, prev_lname))
            prev_lname = lname
            linear_idx += 1

    input_acts = layer_acts["input"]  # (N, input_dim)

    for lname, prev_lname in layer_order:
        curr_acts  = layer_acts[lname]           # (N, out_neurons)
        prev_acts  = layer_acts[prev_lname]      # (N, prev_neurons)

        # Pearson r: curr neurons (cols) vs input features
        r_input = pearson_r_matrix(input_acts, curr_acts)  # (input_dim, out_neurons)
        # Pearson r: curr neurons vs previous layer neurons
        r_layer = pearson_r_matrix(prev_acts,  curr_acts)  # (prev_neurons, out_neurons)

        prev_names = [n for (_, _, n) in layer_neuron_names[prev_lname]]

        for ni, nid, _ in layer_neuron_names[lname]:
            if graph[nid]["type"] in ("input", "output"):
                continue

            node = graph[nid]

            # ── input correlations ──────────────────────────────────
            ic = r_input[:, ni]           # shape (input_dim,)
            top_in = sorted(range(len(ic)), key=lambda i: -abs(ic[i]))[:top_k_input]
            node["input_corr"] = [
                {"name": input_names[i], "r": float(ic[i])}
                for i in top_in
            ]

            # ── layer (prev) correlations ───────────────────────────
            lc = r_layer[:, ni]           # shape (prev_neurons,)
            top_lc = sorted(range(len(lc)), key=lambda i: -abs(lc[i]))[:top_k_layer]
            node["layer_corr"] = [
                {"name": prev_names[i], "r": float(lc[i])}
                for i in top_lc
            ]

            # ── class affinity (classification only) ────────────────
            if task == "classification":
                acts_1d = curr_acts[:, ni]
                class_means = {
                    cls: float(acts_1d[y == ci].mean()) if (y == ci).any() else 0.0
                    for ci, cls in enumerate(class_names)
                }
                best_cls = max(class_means, key=lambda c: class_means[c])
                node["class_affinity"] = best_cls
                node["class_means"]    = class_means
            else:
                node["class_affinity"] = None
                node["class_means"]    = {}

            # ── update name_hint deterministically ──────────────────
            # Use top input correlation: sign + cleaned feature name
            if node["input_corr"]:
                top  = node["input_corr"][0]
                sign = "+" if top["r"] >= 0 else "-"
                # strip units in parens, e.g. "petal length (cm)" -> "petal_length"
                fname = re.sub(r"\s*\(.*?\)", "", top["name"]).strip()
                fname = re.sub(r"\s+", "_", fname)[:18]
                node["name_hint"] = f"{sign}{fname}"
            # If class task, also append class tag
            if task == "classification" and node["class_affinity"]:
                node["name_hint"] += f"/{node['class_affinity']}"


print("\n  [annotate] computing deterministic correlation annotations...", flush=True)
annotate_graph_deterministic(
    graph, levels, X, model, act_np,
    HIDDEN_DIMS, INPUT_DIM, INPUT_NAMES,
    CLASS_NAMES, TASK, y if TASK == "classification" else None,
    top_k_input=3, top_k_layer=2
)

# ── second pass: fix layer_corr names ────────────────────────────────────────
# layer_corr entries store raw node IDs (e.g. "hidden1:2", "input:0").
# Now that all name_hints are finalised, resolve to display names.
# input:N  ->  INPUT_NAMES[N]   (feature name, not graph name_hint which is also the feature name)
# other    ->  graph[nid]["name_hint"]
for nid, node in graph.items():
    if "layer_corr" not in node:
        continue
    for entry in node["layer_corr"]:
        key = entry["name"]
        if key in graph:
            entry["name"] = graph[key]["name_hint"]

# ── third pass: deduplicate colliding name_hints ─────────────────────────────
# Two distinct neurons can legitimately share a name (same top correlated
# feature, same sign, same class affinity).  Append #<neuron_index> only to
# the colliding entries so the reader can tell them apart, e.g.:
#   -petal_length/0#1   and   -petal_length/0#3
seen_names: dict = {}   # name -> first nid that used it
for nid, node in graph.items():
    if node["type"] in ("input", "output"):
        continue
    hint = node["name_hint"]
    if hint not in seen_names:
        seen_names[hint] = nid
    else:
        # collision — tag BOTH the original and the new one with #neuron_idx
        orig_nid  = seen_names[hint]
        orig_node = graph[orig_nid]
        if not re.search(r"#\d+$", orig_node["name_hint"]):
            orig_node["name_hint"] = f"{orig_node['name_hint']}#{orig_node['neuron']}"
        node["name_hint"] = f"{hint}#{node['neuron']}"
        # keep tracking under the untagged base so further collisions also get tagged
        seen_names[hint] = nid   # update so next collision compares against latest

# after dedup, re-resolve layer_corr names once more (they may point to nodes
# whose name_hint just gained a #N suffix)
for nid, node in graph.items():
    if "layer_corr" not in node:
        continue
    for entry in node["layer_corr"]:
        key = entry["name"]
        if key in graph:
            entry["name"] = graph[key]["name_hint"]

print("  [annotate] done\n", flush=True)


# ── 4c. LEGACY OLLAMA AUTONAME (optional, --autoname flag) ───────────────────
def ollama_name(nid, node, graph):
    import json as _json, http.client, urllib.parse

    dep_lines = []
    for dep in node["deps"]:
        dep_name = graph[dep["node"]]["name_hint"]
        pol      = "+" if dep["polarity"] == 1 else "-"
        dep_lines.append(f"  {pol} {dep_name} ({dep['weight']*100:.1f}%)")

    deps_str  = "\n".join(dep_lines) if dep_lines else "  (no dependencies)"
    threshold = node["threshold"]

    prompt = (
        f"Give a snake_case name (2-4 words) for this neuron based on what activates it.\n\n"
        f"EXAMPLE:\n+ petal_length (60%) | - sepal_width (40%) | threshold +1.2\n-> long_petal_narrow_sepal\n\n"
        f"NOW NAME THIS:\nthreshold {threshold:+.3f}\n{deps_str}\n->\n"
    )

    payload = _json.dumps({
        "model":  OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 32}
    }).encode()

    try:
        parsed = urllib.parse.urlparse(OLLAMA_URL)
        conn   = http.client.HTTPConnection(parsed.netloc, timeout=30)
        conn.request("POST", parsed.path, body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise Exception(f"HTTP {resp.status}: {body.decode()[:200]}")
        raw   = _json.loads(body)["message"]["content"].strip()
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        candidate = lines[-1] if lines else ""
        name = re.sub(r"[^\w\s]", "", candidate).strip().replace(" ", "_").lower()
        name = name.lstrip("0123456789_")
        return name if name else node["name_hint"]
    except Exception as e:
        print(f"  [autoname] ollama error for {nid}: {e}", flush=True)
        return node["name_hint"]


def autoname_graph(graph, levels):
    ordered = sorted(
        [nid for nid, n in graph.items() if n["type"] not in ("input", "output")],
        key=lambda nid: levels[nid]
    )
    seen_names = {}
    for i, nid in enumerate(ordered):
        node = graph[nid]
        name = ollama_name(nid, node, graph)
        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 1
        graph[nid]["name_hint"] = name
        print(f"  [autoname] {i+1}/{len(ordered)}  {nid} -> {name}", flush=True)


if AUTONAME:
    print("\n  [autoname] naming neurons via ollama (overrides deterministic names)...", flush=True)
    autoname_graph(graph, levels)
    print("  [autoname] done\n", flush=True)


# ── 5. VALIDATION ─────────────────────────────────────────────────────────────

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


# ── 6. OUTPUT ─────────────────────────────────────────────────────────────────

def polarity_str(p):
    return "+" if p == 1 else "-"

def fmt_r(r):
    sign = "+" if r >= 0 else "-"
    return f"r={r:+.3f}"

lines = []
w = lines.append

w("=" * 70)
w("  SDHCE — Symbolic Distillation via Hierarchical Concept Extraction")
w(f"  Dataset   : {dataset_path}")
w(f"  Arch      : {INPUT_DIM} -> {' -> '.join(str(h) for h in HIDDEN_DIMS)} -> {OUTPUT_DIM}")
w(f"  Activation: {ACTIVATION.upper()}   |   tau_percentile: {TAU_PERCENTILE}")
w("=" * 70)
w("")
if TASK == "classification":
    w(f"  {train_metric_label}: {train_metric*100:.1f}%")
else:
    w(f"  {train_metric_label}: {train_metric:.4f}")
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
    w("-" * 70)
    w(f"  LEVEL {level}  [{level_type}S]")
    w("-" * 70)
    for nid in nodes:
        node = graph[nid]
        w(f"\n  [{node['name_hint']}]  (threshold = {node['threshold']:+.3f})")

        # ── weight-based deps (structural) ──────────────────────────
        w("   → weight deps:")
        if not node["deps"]:
            w("      (none above tau)")
        for dep in node["deps"]:
            lbl = graph[dep["node"]].get("name_hint", dep["node"])
            pol = polarity_str(dep["polarity"])
            w(f"      {pol} {lbl:<26} (contributes {dep['weight']*100:.1f}%)")

        # ── input correlations ──────────────────────────────────────
        if "input_corr" in node and node["input_corr"]:
            w("   → input correlations (Pearson r with raw features):")
            for ic in node["input_corr"]:
                bar_len = int(abs(ic["r"]) * 20)
                bar     = ("█" * bar_len).ljust(20)
                direction = "high" if ic["r"] > 0 else "low"
                w(f"      {ic['name']:<22} {fmt_r(ic['r'])}  [{bar}]  fires when {direction}")

        # ── layer correlations ──────────────────────────────────────
        if "layer_corr" in node and node["layer_corr"]:
            w("   → layer correlations (Pearson r with prev-layer neurons):")
            for lc in node["layer_corr"]:
                bar_len = int(abs(lc["r"]) * 20)
                bar     = ("█" * bar_len).ljust(20)
                w(f"      {lc['name']:<22} {fmt_r(lc['r'])}  [{bar}]")

        # ── class affinity ──────────────────────────────────────────
        if node.get("class_affinity") and TASK == "classification":
            w(f"   → class affinity: {node['class_affinity']}")
            if node.get("class_means"):
                for cls, mean in sorted(node["class_means"].items(), key=lambda x: -x[1]):
                    bar_len = max(0, int((mean + 2) * 4))   # shift for possibly negative means
                    bar     = ("█" * min(bar_len, 20)).ljust(20)
                    w(f"      class {cls:<10}  mean_act={mean:+.3f}  [{bar}]")

    w("")

w("  NOTE: name_hints are auto-derived from top input correlation.")
w("        A human translator should verify and rename as needed.")
w("")
w("=" * 70)
w("  VALIDATION: symbolic hierarchy vs original network")
w("=" * 70)
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
w("=" * 70)
w("")
w("  HOW TO READ THIS OUTPUT")
w("  ───────────────────────")
w("")
w("  STRUCTURE")
w("  The network is distilled into a hierarchy of named nodes.")
w("  Each level builds on the one below:")
w("    Level 1  ATOMS    — neurons directly over raw input features")
w("    Level 2+ CONCEPTS — neurons over earlier learned atoms/concepts")
w("    Level N  OUTPUTS  — final class/regression logits")
w("")
w("  NODE HEADER")
w("    [+petal_width/2]  (threshold = +1.853)")
w("    └─ name_hint  : sign + top correlated input feature + /class_affinity")
w("       sign +     : neuron fires when that feature is HIGH")
w("       sign -     : neuron fires when that feature is LOW")
w("       /N         : neuron's mean activation is highest for class N")
w("       #N suffix  : disambiguates two neurons with the same auto-label")
w("                    (N = neuron index within its layer)")
w("    threshold     : neuron 'switches on' when weighted input sum > this")
w("")
w("  WEIGHT DEPS  (structural — from trained weights)")
w("    Literal wiring: which upstream nodes feed this one, their sign")
w("    (+ excitatory / - inhibitory) and normalised % contribution.")
w("")
w("  INPUT CORRELATIONS  (empirical — Pearson r vs raw features)")
w("    How linearly this neuron's activation tracks each input feature")
w("    across all training samples.  r≈±1 = near-perfect tracking,")
w("    r≈0 = independent.  Validates the name_hint and reveals")
w("    unexpected feature dependencies.")
w("")
w("  LAYER CORRELATIONS  (empirical — Pearson r vs prev-layer neurons)")
w("    Which upstream learned concept this neuron most closely refines.")
w("    High r = 'rescaled version of that concept'.")
w("    Low r  = 'genuine composition of multiple upstream signals'.")
w("")
w("  CLASS AFFINITY BARS  (classification only)")
w("    Mean post-activation per class.  Large positive for one class")
w("    + near-zero for others = clean class detector.")
w("    Similar means across classes = shared feature extractor.")
w("")
w("  VALIDATION BLOCK")
w("    Symbolic accuracy : re-runs the extracted graph as plain arithmetic")
w("    (no PyTorch) on the training set.  Should equal network accuracy.")
w("    Perfect agreement = lossless symbolic transcription, no approximation.")
w("")
w("  WORKFLOW FOR HUMAN TRANSLATION")
w("  1. Level 1 ATOMS: read input_corr to see what raw signal each atom")
w("     tracks; cross-check sign against weight deps.")
w("  2. Rename any atom whose auto-label is wrong or ambiguous.")
w("  3. Level 2+ CONCEPTS: read layer_corr to see which atoms it composes.")
w("     Concept = weighted conjunction/disjunction of those atoms.")
w("  4. OUTPUT level: + dep = concept votes FOR that class,")
w("                   - dep = concept votes AGAINST it.")
w("  5. Build IF-THEN rules using threshold:")
w("     IF weighted_sum(deps) > threshold THEN neuron is active.")
w("")
w("=" * 70)

output_text = "\n".join(lines)
print(output_text)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output_text + "\n")

print(f"\n  [saved to {output_path}]")
