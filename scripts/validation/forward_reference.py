#!/usr/bin/env python3
"""Generate independent Qwen3 forward fixtures without calling Dolittle.

Requires pinned verification dependencies (CUDA by default, CPU optional);
actual weights are verified
against models-8b.json, never fetched. Outputs belong in an ignored fixture cache.
Tiny models deliberately exercise projection tails, GQA and Hq*d != D.

Stock Transformers BF16 eager attention adds intermediate score/probability
roundings absent from the accepted engine contract. The independent dense
PyTorch override below promotes attention operands and rounds only its output.
The oracle retains FP32 accumulation and uses a fixed one-row projection call
and one-query causal attention call, so batch/sequence shape cannot select a
new reduction for an existing row. RMSNorm/RoPE/SiLU remain unchanged. Exact
prefix and peer-batch checks must pass before a fixture is written. Stock eager
sensitivity is recorded separately. Acceptance budgets remain fixed;
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
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
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


def invariant_linear(module, inputs):
    """FP32 GEMV with a canonical shape, independent of outer row count.

    Cast one weight matrix per call; each vector uses [1,I] times [O,I].
    Rounding to model storage happens after the completed FP32 dot product.
    This is an offline oracle: repeated weight reads trade speed for a fixed
    library reduction shape, without copying or importing engine kernels.
    """
    weight = module.weight.float().contiguous()
    bias = None if module.bias is None else module.bias.float()
    flat = inputs.reshape(-1, inputs.shape[-1]).float()
    outputs = [torch.nn.functional.linear(row[None, :].contiguous(), weight, bias)
               for row in flat]
    return torch.cat(outputs, dim=0).to(inputs.dtype).reshape(*inputs.shape[:-1], weight.shape[0])


def invariant_attention(module, query, key, value, attention_mask,
                        scaling, dropout=0.0, **_kwargs):
    """Dense FP32 attention, evaluated on each query's actual causal prefix.

    Full-forward oracle only: Q/K/V share the sequence extent. For each sample
    and absolute position p, contiguous operands have shapes [1,H,1,d] and
    [1,H,p+1,d]. Future/padded positions cannot change that query's reduction
    extent or matrix strides. Scores, probabilities and value sums stay FP32;
    only the completed output rounds to model storage. No FP64 arithmetic.
    """
    if dropout != 0.0 or module.training or query.shape[2] != key.shape[2]:
        raise ValueError("oracle requires full-forward evaluation without dropout")
    key = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
    value = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
    batches = []
    for batch in range(query.shape[0]):
        outputs = []
        for position in range(query.shape[2]):
            q = query[batch:batch+1, :, position:position+1].float().contiguous()
            k = key[batch:batch+1, :, :position+1].float().contiguous()
            v = value[batch:batch+1, :, :position+1].float().contiguous()
            scores = torch.matmul(q, k.transpose(2, 3)) * scaling
            if attention_mask is not None:
                mask_batch = batch if attention_mask.shape[0] > 1 else 0
                mask_row = position if attention_mask.shape[2] > 1 else 0
                scores = scores + attention_mask[mask_batch:mask_batch+1, :, mask_row:mask_row+1,
                                                :position+1].float()
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
            outputs.append(torch.matmul(probabilities, v).to(query.dtype))
        batches.append(torch.cat(outputs, dim=2))
    return torch.cat(batches, dim=0).transpose(1, 2).contiguous(), None


def verify_oracle_invariance(model, generation, dtype):
    """Reject fixtures whose shared storage bits depend on prefix or peers.

    This is a measured guarantee for the fixture/environment, not all hardware.
    Check multiple prefix lengths and put the trace before and after an unrelated
    peer. Hashes cover original stored bytes, including the sign of zero.
    """
    full = generation["reference"]
    tokens = full["token_ids"]
    checks = []
    def require_equal(actual, count, label):
        for expected, observed in zip([full["logits"], *full["captures"]],
                                      [actual["logits"], *actual["captures"]], strict=True):
            if (expected["width"] != observed["width"] or
                    expected["row_sha256"][:count] != observed["row_sha256"]):
                raise ValueError(f"FP32 oracle failed exact shape invariance: {label}")
        checks.append({"case": label, "tokens": count, "equal_stored_bytes": True})
    for count in sorted({1, generation["prompt_tokens"], len(tokens)-1}):
        prefix = evaluate(model, "oracle-prefix", tokens[:count], dtype, all_rows=True)
        require_equal(prefix, count, f"prefix-{count}")
    peer = [(token+5) % model.config.vocab_size for token in tokens]
    for sample, batch in ((0, [tokens, peer]), (1, [peer, tokens])):
        result = evaluate(model, "oracle-peer-batch", tokens, dtype, all_rows=True,
                          batch_tokens=batch, sample=sample)
        require_equal(result, len(tokens), f"peer-batch-slot-{sample}")
    return {"checks": checks, "all_shared_logits_and_captures_equal": True}


def environment(device):
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "safetensors": importlib.metadata.version("safetensors"),
        "cuda_runtime": torch.version.cuda, "execution_device": device,
        "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "compute_capability": list(torch.cuda.get_device_capability()) if device == "cuda" else None,
        "driver": subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ], text=True).strip() if device == "cuda" else None,
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
    stored = tensor[0, positions].detach().contiguous().cpu()
    values = stored.float().reshape(-1).tolist()
    raw_rows = stored.view(torch.uint8).reshape(len(positions), -1).numpy()
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in raw_rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite reference output")
    return {"positions": list(positions), "width": tensor.shape[-1],
            "original_storage_type": dtype, "values": values, "row_sha256": hashes,
            "tolerance": tolerance(values, dtype, blocks)}


def evaluate(model, name, tokens, dtype, all_rows=False, batch_tokens=None, sample=0):
    blocks = model.config.num_hidden_layers
    observed = sorted({0, blocks // 2, blocks - 1} | ({24} if blocks > 24 else set()))
    positions = list(range(len(tokens))) if all_rows else sorted({0, len(tokens) // 2, len(tokens) - 1})
    logit_positions = list(range(len(tokens))) if all_rows else sorted({0, len(tokens) - 1})
    captured, handles = {}, []
    for block in observed:
        def before(_module, args, block=block):
            captured[(block, "block_input")] = rows(args[0][sample:sample+1], positions, dtype, blocks)

        def after(_module, _args, result, block=block):
            value = result[0] if isinstance(result, tuple) else result
            captured[(block, "block_output")] = rows(value[sample:sample+1], positions, dtype, blocks)

        handles.extend([model.model.layers[block].register_forward_pre_hook(before),
                        model.model.layers[block].register_forward_hook(after)])
    try:
        input_ids = torch.tensor(batch_tokens if batch_tokens is not None else [tokens],
                                 dtype=torch.long, device=next(model.parameters()).device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, return_dict=True)
        logits = rows(output.logits[sample:sample+1], logit_positions, dtype, blocks)
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
                or item["path"] in ("model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")]
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


def generation_case(model, dtype, checkpoint):
    """Reference greedily emits four IDs, then evaluates the complete consumed trace.

    Full-prefix recomputation is independent of Dolittle's cached path. Every
    position has reference logits and semantic captures, including the final ID.
    Tokenizer vectors are reference facts, not calls to the implementation under test.
    """
    text_data = None
    if checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        messages = [{"role": "user", "content": "What is 2 + 2?"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
        import random
        rng = random.Random(SEED)
        corpus = ["", "Hello world!", "I'm I'M we're THEY'VE I'd", "1234567890 ٢٣４", "e\u0301 café Å A\u030a", "中文 日本語 한국어", "👩‍💻 🦋 🇬🇧", "a\x00b", "\r\n\t  x  ", "<think>yes</think><|im_end|>"]
        for _ in range(300):
            cps = [rng.randrange(0x110000) for _ in range(30)]
            corpus.append("".join(chr(c) for c in cps if not 0xd800 <= c <= 0xdfff))
        # ASCII overlaps, repeated BPE pairs and every declared added token.
        corpus += ["a" * n + " bbcabc" for n in range(1,70)]
        corpus += list(tokenizer.added_tokens_decoder[k].content for k in sorted(tokenizer.added_tokens_decoder))
        vectors = [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)} for text in corpus]
        prompts = []
        for enabled in (True, False):
            for system in (None, "", "Be concise."):
                for user in ("What is 2 + 2?", "e\u0301 <think> 123\n"):
                    msg = ([] if system is None else [{"role":"system", "content":system}]) + [{"role":"user", "content":user}]
                    kwargs = dict(add_generation_prompt=True, enable_thinking=enabled)
                    prompts.append({"system":system, "user":user, "thinking":enabled,
                        "formatted":tokenizer.apply_chat_template(msg, tokenize=False, **kwargs),
                        "ids":tokenizer.apply_chat_template(msg, tokenize=True, **kwargs)})
        text_data = {"tokenizers_version":importlib.metadata.version("tokenizers"), "vectors":vectors, "prompts":prompts}
    else:
        prompt = [3, 10, 17, 24, 31]
    tokens = list(prompt)
    with torch.inference_mode():
        for _ in range(4):
            inputs = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
            tokens.append(int(logits[0,-1].argmax()))
    return {"prompt_tokens":len(prompt), "max_new_tokens":4, "text":text_data,
            "reference":evaluate(model,"generation",tokens,dtype,all_rows=True)}


def generate(destination, kind, actual_checkpoint, manifest_path, env, device):
    destination.mkdir(parents=True, exist_ok=True)
    actual = kind == "qwen3-8b-bf16"
    dtype = "float32" if kind == "tiny-fp32" else "bfloat16"
    torch_dtype = getattr(torch, dtype)
    if actual:
        artifacts = link_checkpoint(actual_checkpoint, destination, manifest_path)
        print("Verified pinned Qwen3-8B config/index/shards", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            actual_checkpoint, torch_dtype=torch_dtype, attn_implementation="eager",
            local_files_only=True, trust_remote_code=False, device_map=device).eval().requires_grad_(False)
        repository, revision = REPOSITORY, REVISION
        tokens, prefix_length = [151644, 8948, 198, 9707, 151645], 3
    else:
        model = tiny_model(torch_dtype)
        model.save_pretrained(destination, safe_serialization=True)
        artifacts = sorted([artifact(path) for path in destination.iterdir()
                            if path.name == "config.json" or path.suffix == ".safetensors"],
                           key=lambda item: item["relative_path"])
        model = model.to(device)
        repository = "dolittle/independent-tiny-qwen3"
        revision = f"seed-{SEED}-{dtype}-generator-sha256-{digest(Path(__file__))}"
        tokens, prefix_length = [(index*7+3) % 37 for index in range(17)], 5
    print(f"Loaded {kind}", flush=True)
    stock_function = modeling_qwen3.eager_attention_forward
    stock_linear = torch.nn.Linear.forward
    try:
        stock_case = evaluate(model, "stock-eager-sensitivity", tokens, dtype)
        modeling_qwen3.eager_attention_forward = invariant_attention
        torch.nn.Linear.forward = invariant_linear
        generation = generation_case(model, dtype, actual_checkpoint if actual else None)
        invariance = verify_oracle_invariance(model, generation, dtype)
        cases = [evaluate(model, "full", tokens, dtype),
                 evaluate(model, "prefix", tokens[:prefix_length], dtype),
                 evaluate(model, "alternate", [(token+5) % model.config.vocab_size for token in tokens], dtype)]
    finally:
        modeling_qwen3.eager_attention_forward = stock_function
        torch.nn.Linear.forward = stock_linear
    source = Path(inspect.getsourcefile(modeling_qwen3))
    provenance = {
        "implementation_revision": f"transformers-{VERSION};modeling_qwen3.py-sha256-{digest(source)}",
        "generator_sha256": digest(Path(__file__)),
        "environment_identity": json.dumps(env, sort_keys=True), "environment": env,
        "attention_backend": "eager-fp32-per-query-causal-prefix",
        "numerical_policy": "BF16/FP32 stored model; FP32 one-row projection calls and per-query causal attention; TF32 off; stock Transformers RMSNorm/RoPE/SiLU; no FP64 accumulation",
        "attention_override_source": inspect.getsource(invariant_attention),
        "linear_override_source": inspect.getsource(invariant_linear),
        "oracle_shape_invariance": invariance,
        "stock_eager_sensitivity": sensitivity(cases[0], stock_case),
        "seed": None if actual else SEED,
    }
    def maximum_bound(bounds):
        return {key: max(bound[key] for bound in bounds)
                for key in ("max_absolute_error", "max_relative_l2")}
    logits_bounds = [case["logits"]["tolerance"] for case in [*cases, generation["reference"]]]
    observation_bounds = [capture["tolerance"] for case in [*cases, generation["reference"]] for capture in case["captures"]]
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
        "cases": cases, "generation": generation,
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
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", choices=["tiny", "actual", "all"], default="all")
    parser.add_argument("--actual-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("models-8b.json"))
    args = parser.parse_args()
    if (importlib.metadata.version("transformers") != VERSION or
            importlib.metadata.version("tokenizers") != "0.22.2"):
        raise RuntimeError("requires pinned Transformers 4.57.3 and tokenizers 0.22.2")
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("requires CUDA BF16 support")
    if args.kind in ("actual", "all") and args.actual_checkpoint is None:
        parser.error("--actual-checkpoint is required for actual fixtures")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    env = environment(args.device)
    kinds = (["tiny-fp32", "tiny-bf16"] if args.kind in ("tiny", "all") else [])
    if args.kind in ("actual", "all"):
        kinds.append("qwen3-8b-bf16")
    for kind in kinds:
        generate(args.output / kind, kind, args.actual_checkpoint, args.manifest, env, args.device)


if __name__ == "__main__":
    main()
