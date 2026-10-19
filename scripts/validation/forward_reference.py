#!/usr/bin/env python3
"""Generate independent Qwen3 forward fixtures without calling Dolittle.

Requires pinned verification dependencies and CUDA; actual weights are verified
against models-8b.json, never fetched. Outputs belong in an ignored fixture cache.
Tiny models deliberately exercise projection tails, GQA and Hq*d != D.

Stock Transformers BF16 eager attention adds intermediate score/probability
roundings absent from the accepted engine contract. The independent dense
PyTorch override below promotes attention operands to FP32 and rounds only its
output. All other Transformers math remains unchanged. Stock eager sensitivity
is recorded separately. Acceptance budgets are fixed before custom execution;
they are numerical tolerances, not semantic or formal forward-error guarantees.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import torch
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3

SEED = 20261018
REPOSITORY = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
VERSION = "4.57.3"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fp32_eager_attention(module, query, key, value, attention_mask,
                         scaling, dropout=0.0, **_kwargs):
    """Independent dense equation: materialized FP32 scores and probabilities."""
    if dropout != 0.0 or module.training:
        raise ValueError("reference requires evaluation without dropout")
    key = modeling_qwen3.repeat_kv(key, module.num_key_value_groups).float()
    value = modeling_qwen3.repeat_kv(value, module.num_key_value_groups).float()
    scores = torch.matmul(query.float(), key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :key.shape[-2]].float()
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(probabilities, value).to(query.dtype)
    return output.transpose(1, 2).contiguous(), probabilities


def environment():
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "safetensors": importlib.metadata.version("safetensors"),
        "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "driver": subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ], text=True).strip(),
        "tf32": False, "bf16_reduced_precision_reduction": False,
    }


def tolerance(values, dtype, blocks):
    peak = max((abs(value) for value in values), default=0.0)
    if dtype == "float32":
        factor, floor, relative = 2e-4, 1e-5, 2e-4
    elif blocks <= 3:
        factor, floor, relative = 0.03125, 1e-4, 0.02
    else:
        factor, floor, relative = 0.0625, 1e-4, 0.03
    return {"max_absolute_error": max(floor, peak * factor),
            "max_relative_l2": relative}


def rows(tensor, positions, dtype, blocks):
    values = tensor[0, positions].detach().float().cpu().reshape(-1).tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite reference output")
    return {"positions": list(positions), "width": tensor.shape[-1],
            "original_storage_type": dtype, "values": values,
            "tolerance": tolerance(values, dtype, blocks)}


def evaluate(model, name, tokens, dtype):
    blocks = model.config.num_hidden_layers
    observed = sorted({0, blocks // 2, blocks - 1} | ({24} if blocks > 24 else set()))
    positions = sorted({0, len(tokens) // 2, len(tokens) - 1})
    logit_positions = sorted({0, len(tokens) - 1})
    captured, handles = {}, []
    for block in observed:
        def before(_module, args, block=block):
            captured[(block, "block_input")] = rows(args[0], positions, dtype, blocks)

        def after(_module, _args, result, block=block):
            value = result[0] if isinstance(result, tuple) else result
            captured[(block, "block_output")] = rows(value, positions, dtype, blocks)

        handles.extend([model.model.layers[block].register_forward_pre_hook(before),
                        model.model.layers[block].register_forward_hook(after)])
    try:
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, return_dict=True)
        logits = rows(output.logits, logit_positions, dtype, blocks)
    finally:
        for handle in handles:
            handle.remove()
    captures = []
    for block in observed:
        for boundary in ("block_input", "block_output"):
            captures.append({"boundary": boundary, "block": block,
                             **captured[(block, boundary)]})
    return {"name": name, "token_ids": list(tokens), "logit_positions": logit_positions,
            "logits": logits, "captures": captures}


def error(reference, actual):
    if len(reference) != len(actual):
        raise ValueError("sensitivity shape mismatch")
    squared_difference = math.fsum((a-b)**2 for a, b in zip(reference, actual))
    squared_reference = math.fsum(a*a for a in reference)
    return {"max_absolute_error": max((abs(a-b) for a, b in zip(reference, actual)), default=0.0),
            "relative_l2": math.sqrt(squared_difference / squared_reference)
            if squared_reference else None}


def sensitivity(policy, stock):
    result = {"logits": error(policy["logits"]["values"], stock["logits"]["values"])}
    for expected, actual in zip(policy["captures"], stock["captures"]):
        key = f'block_{expected["block"]}_{expected["boundary"]}'
        result[key] = error(expected["values"], actual["values"])
    return result


def tiny_model(dtype):
    torch.manual_seed(SEED)
    config = Qwen3Config(vocab_size=37, hidden_size=24, intermediate_size=40,
                        num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                        rms_norm_eps=1e-6, rope_theta=10000.0, attention_bias=False,
                        attention_dropout=0.0, tie_word_embeddings=False,
                        use_sliding_window=False, sliding_window=None,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=False)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).to(dtype=dtype).eval().requires_grad_(False)


def artifact(path):
    return {"relative_path": path.name, "byte_size": path.stat().st_size,
            "sha256": digest(path)}


def link_checkpoint(source, destination, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    target = next(model for model in manifest["models"] if model["repo"] == REPOSITORY)
    if target["revision"] != REVISION:
        raise ValueError("unexpected target revision")
    selected = [item for item in target["files"] if item["path"] == "config.json"
                or item["path"].endswith(".safetensors")
                or item["path"] == "model.safetensors.index.json"]
    artifacts = []
    for item in selected:
        origin = source / item["path"]
        if origin.stat().st_size != item["bytes"] or digest(origin) != item["sha256"]:
            raise ValueError(f'checkpoint integrity failure: {item["path"]}')
        target_path = destination / item["path"]
        if target_path.exists():
            if not os.path.samefile(origin, target_path):
                raise ValueError(f"refusing to replace existing artifact: {target_path.name}")
        else:
            os.link(origin, target_path)
        artifacts.append({"relative_path": item["path"], "byte_size": item["bytes"],
                          "sha256": item["sha256"]})
    return sorted(artifacts, key=lambda item: item["relative_path"])


def generate(destination, kind, actual_checkpoint, manifest_path, env):
    destination.mkdir(parents=True, exist_ok=True)
    actual = kind == "qwen3-8b-bf16"
    dtype = "float32" if kind == "tiny-fp32" else "bfloat16"
    torch_dtype = getattr(torch, dtype)
    if actual:
        artifacts = link_checkpoint(actual_checkpoint, destination, manifest_path)
        print("Verified pinned Qwen3-8B config/index/shards", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            actual_checkpoint, torch_dtype=torch_dtype, attn_implementation="eager",
            local_files_only=True, trust_remote_code=False, device_map="cuda").eval().requires_grad_(False)
        repository, revision = REPOSITORY, REVISION
        tokens, prefix_length = [151644, 8948, 198, 9707, 151645], 3
    else:
        model = tiny_model(torch_dtype)
        model.save_pretrained(destination, safe_serialization=True)
        artifacts = sorted([artifact(path) for path in destination.iterdir()
                            if path.name == "config.json" or path.suffix == ".safetensors"],
                           key=lambda item: item["relative_path"])
        model = model.to("cuda")
        repository = "dolittle/independent-tiny-qwen3"
        revision = f"seed-{SEED}-{dtype}-generator-sha256-{digest(Path(__file__))}"
        tokens, prefix_length = [(index*7+3) % 37 for index in range(17)], 5
    print(f"Loaded {kind}", flush=True)
    stock_function = modeling_qwen3.eager_attention_forward
    try:
        stock_case = evaluate(model, "stock-eager-sensitivity", tokens, dtype)
        modeling_qwen3.eager_attention_forward = fp32_eager_attention
        cases = [evaluate(model, "full", tokens, dtype),
                 evaluate(model, "prefix", tokens[:prefix_length], dtype),
                 evaluate(model, "alternate", [(token+5) % model.config.vocab_size for token in tokens], dtype)]
    finally:
        modeling_qwen3.eager_attention_forward = stock_function
    source = Path(inspect.getsourcefile(modeling_qwen3))
    provenance = {
        "implementation_revision": f"transformers-{VERSION};modeling_qwen3.py-sha256-{digest(source)}",
        "generator_sha256": digest(Path(__file__)),
        "environment_identity": json.dumps(env, sort_keys=True), "environment": env,
        "attention_backend": "eager-fp32-score-probability-accumulation-override",
        "numerical_policy": "BF16/FP32 stored model; TF32 off; FP32 reductions; stock Transformers RMSNorm/RoPE/SiLU; only attention override widens Q/K/V before eager matmul/softmax and rounds its final output",
        "attention_override_source": inspect.getsource(fp32_eager_attention),
        "stock_eager_sensitivity": sensitivity(cases[0], stock_case),
        "seed": None if actual else SEED,
    }
    def maximum_bound(bounds):
        return {key: max(bound[key] for bound in bounds)
                for key in ("max_absolute_error", "max_relative_l2")}
    logits_bounds = [case["logits"]["tolerance"] for case in cases]
    observation_bounds = [capture["tolerance"] for case in cases for capture in case["captures"]]
    fixture = {
        "schema_version": 1, "repository": repository, "revision": revision,
        "config_artifact": "config.json", "dtype": dtype, "artifacts": artifacts,
        "shards": sorted(item["relative_path"] for item in artifacts if item["relative_path"].endswith(".safetensors")),
        "index": "model.safetensors.index.json" if actual else None,
        "reference": provenance,
        "tolerances": {
            "logits": maximum_bound(logits_bounds), "observations": maximum_bound(observation_bounds),
            "rationale": "Fixed before custom results: FP32 absolute2e-4*each output peak (floor1e-5), relative2e-4. BF16 unit roundoff=2^-8: tiny maximum allowance8u and relative.02; 36-block allowance16u and relative.03; absolute floor1e-4. Per-output bounds prevent scale masking across sites. These are acceptance budgets, not semantic guarantees or formal forward-error proofs. Stock-eager sensitivity is diagnostic and never changes bounds."
        },
        "cases": cases,
    }
    (destination / "fixture.json").write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"bundle": kind, "fixture": str(destination / "fixture.json"),
                      "stock_eager_sensitivity": provenance["stock_eager_sensitivity"],
                      "tolerances": fixture["tolerances"]}), flush=True)
    del model, stock_case, cases
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", choices=["tiny", "actual", "all"], default="all")
    parser.add_argument("--actual-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("models-8b.json"))
    args = parser.parse_args()
    if importlib.metadata.version("transformers") != VERSION:
        raise RuntimeError("requires pinned Transformers 4.57.3")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("requires CUDA BF16 support")
    if args.kind in ("actual", "all") and args.actual_checkpoint is None:
        parser.error("--actual-checkpoint is required for actual fixtures")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    env = environment()
    kinds = (["tiny-fp32", "tiny-bf16"] if args.kind in ("tiny", "all") else [])
    if args.kind in ("actual", "all"):
        kinds.append("qwen3-8b-bf16")
    for kind in kinds:
        generate(args.output / kind, kind, args.actual_checkpoint, args.manifest, env)


if __name__ == "__main__":
    main()
