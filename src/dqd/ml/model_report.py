"""
model_report.py — what the network IS, as a printable table and as files.

Every run folder should explain its own model without anyone opening the
source, so :func:`save_model_report` writes into <run>/models/ :

    model_structure.yaml    the architecture, layer by layer, plus the
                            training hyperparameters (human-readable)
    model_structure.json    the same data, for programs

and :func:`format_text` renders the layer table that is printed during a run.

The layer table comes from a real forward pass on a dummy input, with hooks
on every leaf module — so the output shapes are what the network actually
produces, not what the source suggests it should.

The architecture is identical in every cell of a budget sweep (the network
is fully convolutional: no flatten, no fixed-size layer), which is exactly
what makes the sweep a fair comparison.  One report per run therefore
describes every checkpoint in it.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .grid_model import RayToLinesNet
from .grid_train import training_description
from .ray_peaks import NET_CHANNELS


# ──────────────────────────────────────────────────────────────────────
#  Layer table
# ──────────────────────────────────────────────────────────────────────

def layer_table(net: torch.nn.Module,
                input_hw: Tuple[int, int] = (100, 100)) -> List[Dict]:
    """
    One row per leaf module: name, type, output shape, parameter count.

    Measured by running a dummy batch through the network with forward
    hooks attached.  The network is put in eval mode for the pass (BatchNorm
    cannot handle a batch of one while training) and restored afterwards.
    """
    rows: List[Dict] = []

    def hook(name):
        def fn(module, _inputs, output):
            shape = (tuple(output.shape) if isinstance(output, torch.Tensor)
                     else None)
            rows.append({
                "layer": name,
                "type": type(module).__name__,
                "output_shape": list(shape) if shape else None,
                "n_params": sum(p.numel() for p in module.parameters(
                    recurse=False)),
            })
        return fn

    handles = [m.register_forward_hook(hook(n))
               for n, m in net.named_modules() if not list(m.children())]

    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            net(torch.zeros(1, net.in_channels, *input_hw))
    finally:
        for h in handles:
            h.remove()
        net.train(was_training)
    return rows


def format_text(report: Dict) -> str:
    """The layer table as a printable block."""
    rows = report["layers"]
    w_name = max([len(r["layer"]) for r in rows] + [5])
    w_type = max([len(r["type"]) for r in rows] + [4])
    shape_of = lambda r: ("x".join(str(d) for d in r["output_shape"])
                          if r["output_shape"] else "-")
    w_shape = max([len(shape_of(r)) for r in rows] + [12])

    line = "─" * (w_name + w_type + w_shape + 14)
    out = [
        f"{report['model']['class']} — {report['model']['n_params']:,} parameters",
        f"input {report['input_shape']}  ->  output {report['output_shape']}",
        line,
        f"{'layer':<{w_name}}  {'type':<{w_type}}  "
        f"{'output shape':<{w_shape}}  {'params':>9}",
        line,
    ]
    for r in rows:
        out.append(f"{r['layer']:<{w_name}}  {r['type']:<{w_type}}  "
                   f"{shape_of(r):<{w_shape}}  {r['n_params']:>9,}")
    out += [line, f"{'total':<{w_name + w_type + w_shape + 4}}  "
                  f"{report['model']['n_params']:>9,}"]
    return "\n".join(out)


def build_report(net: Optional[torch.nn.Module] = None,
                 input_hw: Tuple[int, int] = (100, 100),
                 extra: Optional[Dict] = None) -> Dict:
    """
    Everything about the model as one dict: architecture, layer table,
    training hyperparameters, and whatever *extra* the caller adds
    (the run's budgets and checkpoint names, typically).
    """
    net = net if net is not None else RayToLinesNet(in_channels=NET_CHANNELS)
    h, w = input_hw
    rows = layer_table(net, input_hw)
    return {
        "_about": ("The network that produced this run.  It is fully "
                   "convolutional, so every measurement budget in the sweep "
                   "uses this same architecture with this same parameter "
                   "count — accuracy differences come from the measurement, "
                   "not from model capacity.  Layer shapes are measured by a "
                   f"real forward pass on a {h}x{w} grid."),
        "input_shape": f"(batch, {net.in_channels}, {h}, {w})",
        "output_shape": f"(batch, {h}, {w})",
        "model": net.describe(),
        "training": training_description(),
        "layers": rows,
        "n_layers": len(rows),
        **(extra or {}),
    }


# ──────────────────────────────────────────────────────────────────────
#  Minimal YAML writer
# ──────────────────────────────────────────────────────────────────────
#
# PyYAML is not a dependency of this project and the data here is simple
# (nested dicts, lists, scalars), so it is emitted directly.  Every string
# is quoted, which keeps colons, '#' and brackets inside the text harmless.

def _scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{s}"'


def _is_flow(v) -> bool:
    """A non-empty list of scalars — short enough to keep on one line."""
    return (isinstance(v, list) and bool(v)
            and all(not isinstance(x, (dict, list)) for x in v))


def _flow(v) -> str:
    return "[" + ", ".join(_scalar(x) for x in v) + "]"


def to_yaml(obj, indent: int = 0) -> str:
    """Dump nested dicts/lists/scalars as readable YAML."""
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return pad + "{}\n"
        out = ""
        for k, v in obj.items():
            if _is_flow(v):
                out += f"{pad}{k}: {_flow(v)}\n"
            elif isinstance(v, (dict, list)) and v:
                out += f"{pad}{k}:\n{to_yaml(v, indent + 1)}"
            elif isinstance(v, (dict, list)):
                out += f"{pad}{k}: {'{}' if isinstance(v, dict) else '[]'}\n"
            else:
                out += f"{pad}{k}: {_scalar(v)}\n"
        return out
    if isinstance(obj, list):
        if not obj:
            return pad + "[]\n"
        if _is_flow(obj):
            return pad + _flow(obj) + "\n"
        out = ""
        for x in obj:
            body = (to_yaml(x, indent + 1) if isinstance(x, (dict, list))
                    else "  " * (indent + 1) + _scalar(x) + "\n")
            lines = body.rstrip("\n").split("\n")
            # "- " is exactly one indent level wide, so the remaining lines
            # of the item already line up under it.
            lines[0] = pad + "- " + lines[0].lstrip()
            out += "\n".join(lines) + "\n"
        return out
    return f"{pad}{_scalar(obj)}\n"


# ──────────────────────────────────────────────────────────────────────
#  The one call the scripts make
# ──────────────────────────────────────────────────────────────────────

def save_model_report(out_dir: str,
                      net: Optional[torch.nn.Module] = None,
                      input_hw: Tuple[int, int] = (100, 100),
                      extra: Optional[Dict] = None,
                      quiet: bool = False) -> Dict:
    """
    Write model_structure.yaml + model_structure.json into *out_dir* and
    print the layer table.  Returns the report dict.
    """
    import json
    import os

    report = build_report(net, input_hw, extra)
    os.makedirs(out_dir, exist_ok=True)

    # Explicit UTF-8: the text carries em-dashes, and on Windows the default
    # would be cp1252 — which YAML readers, that assume UTF-8, misparse.
    yaml_path = os.path.join(out_dir, "model_structure.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Model structure for this run — see _about below.\n")
        f.write(to_yaml(report))

    json_path = os.path.join(out_dir, "model_structure.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not quiet:
        print(format_text(report))
        print(f"\n  wrote {os.path.abspath(yaml_path)}")
        print(f"  wrote {os.path.abspath(json_path)}")
    return report
