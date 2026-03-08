import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import csv

# ── USAGE ─────────────────────────────────────────────────────────────────────
# python sdhce.py dataset.csv hyperparams.txt output.txt [--autoname] [--ollama-model MODEL] [--ollama-url URL] # the last 3 arguments NOT reccomended to use
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

print("Usage: python sdhce.py dataset.csv hyperparams.txt output.txt")
dataset_path, hyperparam_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
extra_args     = sys.argv[4:]
AUTONAME       = "--autoname" in extra_args
OLLAMA_MODEL   = next((extra_args[i+1] for i, a in enumerate(extra_args) if a == "--ollama-model"), "llama3.2")
OLLAMA_URL     = next((extra_args[i+1] for i, a in enumerate(extra_args) if a == "--ollama-url"),   "http://localhost:11434/api/chat")


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
    # target_cols = "last:N" or "i,j,k"
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
        OUTPUT_DIM = len(CLASS_NAMES)  # override hyperparam — must match num classes
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

    # Collect (W, b, name) for every Linear layer in order
    layer_specs = []
    linear_idx  = 0
    dims        = [input_dim] + hidden_dims + [output_dim]
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

# ── 4b. AUTONAME VIA OLLAMA (optional) ───────────────────────────────────────
# -- NOT RECCOMENDED WITH SMALL MODELS <=3b, larger ones not tested!
def ollama_name(nid, node, graph):
    """
    One call per neuron. Only passes:
    - the node type and threshold
    - its deps with already-resolved names, polarity, and contribution %
    Nothing else.
    """
    import urllib.request, json as _json

    dep_lines = []
    for dep in node["deps"]:
        dep_name = graph[dep["node"]]["name_hint"]
        pol      = "+" if dep["polarity"] == 1 else "-"
        dep_lines.append(f"  {pol} {dep_name} ({dep['weight']*100:.1f}%)")

    deps_str  = "\n".join(dep_lines) if dep_lines else "  (no dependencies)"
    node_type = node["type"]
    threshold = node["threshold"]

    prompt = (
        f"Give a snake_case name (2-4 words) for this neuron based on what activates it.\n"
        f"\n"
        f"EXAMPLE:\n"
        f"+ petal_length (60%) | - sepal_width (40%) | threshold +1.2\n"
        f"-> long_petal_narrow_sepal\n"
        f"\n"
        f"EXAMPLE:\n"
        f"- petal_length (50%) | - petal_width (50%) | threshold -2.0\n"
        f"-> small_petal\n"
        f"\n"
        f"NOW NAME THIS:\n"
        f"threshold {threshold:+.3f}\n"
        f"{deps_str}\n"
        f"->\n"
    )

    payload = _json.dumps({
        "model":  OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 32}
    }).encode()

    try:
        import http.client, urllib.parse
        parsed  = urllib.parse.urlparse(OLLAMA_URL)
        conn    = http.client.HTTPConnection(parsed.netloc, timeout=30)
        conn.request("POST", parsed.path, body=payload,
                     headers={"Content-Type": "application/json"})
        resp    = conn.getresponse()
        body    = resp.read()
        if resp.status != 200:
            raise Exception(f"HTTP {resp.status}: {body.decode()[:200]}")
        raw  = _json.loads(body)["message"]["content"].strip()
        # clean: keep only alphanumeric and underscores, collapse spaces to _
        import re
        # take the last non-empty line — skips any preamble the model adds
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        candidate = lines[-1] if lines else ""
        name = re.sub(r"[^\w\s]", "", candidate).strip().replace(" ", "_").lower()
        # strip leading digits/underscores that sometimes leak in
        name = name.lstrip("0123456789_")
        return name if name else node["name_hint"]
    except Exception as e:
        print(f"  [autoname] ollama error for {nid}: {e}", flush=True)
        return node["name_hint"]


def autoname_graph(graph, levels):
    """Process nodes in level order so deps are always named before dependents."""
    ordered = sorted(
        [nid for nid, n in graph.items() if n["type"] not in ("input", "output")],
        key=lambda nid: levels[nid]
    )
    total     = len(ordered)
    seen_names = {}   # name -> count of times used
    for i, nid in enumerate(ordered):
        node = graph[nid]
        name = ollama_name(nid, node, graph)
        # deduplicate: if name already used, append _2, _3, etc.
        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 1
        graph[nid]["name_hint"] = name
        print(f"  [autoname] {i+1}/{total}  {nid} -> {name}", flush=True)


if AUTONAME:
    print("\n  [autoname] naming neurons via ollama...", flush=True)
    autoname_graph(graph, levels)
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
w("=" * 65)

output_text = "\n".join(lines)
print(output_text)

with open(output_path, "w") as f:
    f.write(output_text + "\n")

print(f"\n  [saved to {output_path}]")
