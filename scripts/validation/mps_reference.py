#!/usr/bin/env python3
"""Freeze a native MPS baseline from the existing independent forward oracle.

Reuse forward_reference.py unchanged. Only device placement, environment
provenance and fixture packaging differ. Existing fixture weights, token
traces, selections and numeric limits are retained exactly. Expected values
come from PyTorch/Transformers on MPS, never from Dolittle. No CPU fallback,
precision promotion, replacement elementary functions or engine math imports.
The output directory must be new; source fixtures are never modified.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess

# These must be set before importing torch. Reject conflicting caller settings.
for variable in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
    if os.environ.get(variable, "0") != "0":
        raise RuntimeError(f"{variable} must be 0 for this reference")
    os.environ[variable] = "0"

# PyTorch 2.7.1 LinearAlgebra.mm::use_metal_mm tests getenv != nullptr,
# not the value: even "0" selects the alternate mm kernel. Unset means off.
# This setting does not control _mps_linear or floating-point bmm; both build
# MPSGraph matrix multiplications directly in this pinned version.
if os.environ.get("PYTORCH_MPS_PREFER_METAL", "0") != "0":
    raise RuntimeError("PYTORCH_MPS_PREFER_METAL must be unset or 0")
os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3 import modeling_qwen3

import forward_reference as oracle

ORACLE_SHA256 = "88e96870ca54e1aa5e1921f668b29b14201b171c3f64a46a66f1b01a957c5d2f"
QWEN_SHA256 = "d96eeea4c000abaa70a7c0277437d0843b5b2d005ab233e0ffcaf860a357f1aa"


def preserve_limits(actual, original):
    """Keep numeric budgets, not just the formula used to derive them."""
    for new, old in zip([actual["logits"], *actual["captures"]],
                        [original["logits"], *original["captures"]], strict=True):
        for key in ("positions", "width", "original_storage_type"):
            if new[key] != old[key]:
                raise ValueError(f"reference selection changed: {key}")
        if (new.get("block"), new.get("boundary")) != (old.get("block"), old.get("boundary")):
            raise ValueError("reference hook identity changed")
        new["tolerance"] = copy.deepcopy(old["tolerance"])
    return actual


def environment():
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "transformers": importlib.metadata.version("transformers"),
        "safetensors": importlib.metadata.version("safetensors"),
        "execution_device": "mps", "device": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "macos": platform.mac_ver()[0], "machine": platform.machine(),
        "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
        "mps_fallback": False, "mps_fast_math": False, "mps_prefer_metal": False,
        "mps_environment": {name: os.environ.get(name) for name in (
            "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH",
            "PYTORCH_MPS_PREFER_METAL")},
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                 for d in importlib.metadata.distributions())),
    }


def verify_single_sequence(model, generation, dtype):
    """Require exact replay/prefixes for Dolittle's one-sequence execution.

    MPS's stock operations can change rounding with a peer batch. Measure that
    separately; it is not a supported execution shape of Dolittle. This changes
    validation scope, not any oracle arithmetic, outputs or numeric limits.
    """
    full = generation["reference"]
    tokens = full["token_ids"]
    checks, peers = [], []

    def compare(actual, count, label):
        outputs = []
        for expected, observed in zip([full["logits"], *full["captures"]],
                                      [actual["logits"], *actual["captures"]], strict=True):
            if expected["width"] != observed["width"]:
                raise ValueError("reference width changed")
            width = expected["width"]
            outputs.append({
                "site": "logits" if "block" not in expected else
                        f"{expected['block']}-{expected['boundary']}",
                "equal_stored_bytes": expected["row_sha256"][:count] == observed["row_sha256"],
                **oracle.error(expected["values"][:count * width], observed["values"]),
            })
        return {"case": label, "tokens": count, "outputs": outputs,
                "equal_stored_bytes": all(x["equal_stored_bytes"] for x in outputs)}

    for label, count in [("repeat-1", len(tokens)), ("repeat-2", len(tokens)),
                         *[(f"prefix-{n}", n) for n in
                           sorted({1, generation["prompt_tokens"], len(tokens)-1})]]:
        result = oracle.evaluate(model, label, tokens[:count], dtype, all_rows=True)
        comparison = compare(result, count, label)
        if not comparison["equal_stored_bytes"]:
            raise ValueError(f"MPS single-sequence reference is not invariant: {label}")
        checks.append(comparison)
    peer = [(token + 5) % model.config.vocab_size for token in tokens]
    for sample, batch in ((0, [tokens, peer]), (1, [peer, tokens])):
        result = oracle.evaluate(model, "peer-batch-diagnostic", tokens, dtype,
                                 all_rows=True, batch_tokens=batch, sample=sample)
        peers.append(compare(result, len(tokens), f"peer-batch-slot-{sample}"))
    return {"scope": "batch-size-one: exact repeatability and prefix invariance",
            "checks": checks, "all_shared_logits_and_captures_equal": True,
            "peer_batch_diagnostics": peers,
            "peer_batches_equal": all(x["equal_stored_bytes"] for x in peers)}


def generate(source, destination, env):
    original = json.loads(source.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    # Reuse exactly the serialized weights, including the tiny models. Do not
    # regenerate their random initialization under a different Python wheel.
    for artifact in original["artifacts"]:
        relative = Path(artifact["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid artifact path")
        path = source.parent / relative
        if (path.stat().st_size != artifact["byte_size"] or
                oracle.digest(path) != artifact["sha256"]):
            raise ValueError(f"checkpoint integrity failure: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
    print(f"Verified checkpoint: {source.parent.name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        source.parent, torch_dtype=getattr(torch, original["dtype"]),
        attn_implementation="eager", device_map="mps", local_files_only=True,
        trust_remote_code=False).eval().requires_grad_(False)
    if any(p.device.type != "mps" or p.dtype != getattr(torch, original["dtype"])
           for p in model.parameters()):
        raise ValueError("reference parameters must retain storage dtype on MPS")
    print("Loaded on MPS; evaluating frozen token inputs", flush=True)
    fixture = copy.deepcopy(original)
    stock_attention = modeling_qwen3.eager_attention_forward
    stock_linear = torch.nn.Linear.forward
    try:
        stock = oracle.evaluate(model, "stock-eager-sensitivity",
                                original["cases"][0]["token_ids"], original["dtype"])
        modeling_qwen3.eager_attention_forward = oracle.invariant_attention
        torch.nn.Linear.forward = oracle.invariant_linear
        fixture["cases"] = []
        for case in original["cases"]:
            observed = oracle.evaluate(model, case["name"], case["token_ids"], original["dtype"])
            fixture["cases"].append(preserve_limits(observed, case))
            print(f"Completed reference case: {case['name']}", flush=True)
        generation = fixture["generation"]
        old_trace = original["generation"]["reference"]
        generation["reference"] = preserve_limits(
            oracle.evaluate(model, old_trace["name"], old_trace["token_ids"],
                            original["dtype"], all_rows=True), old_trace)
        trace = generation["reference"]
        width = trace["logits"]["width"]
        tokens = trace["token_ids"]
        for position in range(generation["prompt_tokens"], len(tokens)):
            logits = trace["logits"]["values"][(position - 1) * width:position * width]
            chosen = max(range(width), key=logits.__getitem__)
            if chosen != tokens[position]:
                raise ValueError(f"MPS greedy trace differs at {position}: {chosen} != {tokens[position]}")
        print("Greedy trace matches; checking exact replay/prefixes and peer diagnostics", flush=True)
        invariance = verify_single_sequence(model, generation, original["dtype"])
    finally:
        modeling_qwen3.eager_attention_forward = stock_attention
        torch.nn.Linear.forward = stock_linear
    torch.mps.synchronize()
    reference = fixture["reference"]
    reference.update({
        "environment": env, "environment_identity": json.dumps(env, sort_keys=True),
        "generator_sha256": oracle.digest(Path(__file__)),
        "shared_oracle_sha256": ORACLE_SHA256,
        "source_fixture_sha256": oracle.digest(source),
        "source_reference": original["reference"],
        "oracle_shape_invariance": invariance,
        "stock_eager_sensitivity": oracle.sensitivity(fixture["cases"][0], stock),
        "linear_override_source": inspect.getsource(oracle.invariant_linear),
        "attention_override_source": inspect.getsource(oracle.invariant_attention),
    })
    # Fresh file plus a separate manifest freezes the completed output before
    # any engine comparison. The source reference and all limits stay intact.
    output = destination / "fixture.json"
    with output.open("x") as stream:
        stream.write(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    manifest = {"fixture_sha256": oracle.digest(output),
                "wrapper_sha256": oracle.digest(Path(__file__)),
                "shared_oracle_sha256": ORACLE_SHA256,
                "source_fixture_sha256": oracle.digest(source),
                "environment": env}
    with (destination / "frozen.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"fixture": str(output), "sha256": manifest["fixture_sha256"]}), flush=True)
    del model, fixture, original, stock
    gc.collect()
    torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("fixtures", nargs="+", type=Path)
    args = parser.parse_args()
    if (torch.__version__ != "2.7.1" or
            importlib.metadata.version("transformers") != "4.57.3" or
            importlib.metadata.version("tokenizers") != "0.22.2" or
            importlib.metadata.version("safetensors") != "0.8.0"):
        raise RuntimeError("requires the pinned MPS reference dependencies")
    if not torch.backends.mps.is_available():
        raise RuntimeError("requires native MPS")
    if oracle.digest(Path(oracle.__file__)) != ORACLE_SHA256:
        raise RuntimeError("shared independent oracle changed")
    if oracle.digest(Path(inspect.getsourcefile(modeling_qwen3))) != QWEN_SHA256:
        raise RuntimeError("pinned Transformers Qwen3 source changed")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    env = environment()
    print(json.dumps(env), flush=True)
    for source in args.fixtures:
        generate(source, args.output / source.parent.name, env)


if __name__ == "__main__":
    main()
