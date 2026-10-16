#!/usr/bin/env python3
"""Reference feasibility probe, not the Dolittle inference implementation.

Loads pinned local Qwen3/NLA safetensors using Transformers and PEFT. One
resident base holds both adapters; target generation disables adapters. Capture
positions identify the token just consumed, whose residual predicts the NEXT
token. Causal replay collects those states without retaining every layer.
AV prompts and normalization come from checkpoint metadata; AR reads the output
of the requested block at the last token. Outputs include raw transcripts,
timing, memory, and matched-versus-swapped reconstruction comparisons. These
comparisons are diagnostics, not calibrated explanation confidence.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    ("arithmetic", "A box has 12 red marbles and 7 blue marbles. I remove 5 red marbles and add 3 blue marbles. How many marbles are in the box now?"),
    ("tracking", "Mara puts a copper key in the kitchen drawer. Later she moves it into her coat pocket. Her coat is hanging by the door. Where is the key now?"),
    ("poetry", "Write a two-line poem about a rabbit. End the first line with habit and the second line with rabbit."),
    ("logic", "All cedar trees in this garden are tall. Some tall trees in this garden are old. Does it follow that some cedar trees in this garden are old? Explain briefly."),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["thinking", "prompt-end"], default="thinking")
    ap.add_argument("--av-template", choices=["card", "training"], default="card",
                    help="card disables AV thinking; training uses the tokenizer default")
    args = ap.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.is_available(), "CUDA GPU required"
    assert torch.cuda.is_bf16_supported(), "BF16 reference requires hardware support"
    manifest = json.loads((args.root / "models.json").read_text())
    paths = {m["repo"]: args.root / "models" / m["repo"].replace("/", "--") for m in manifest["models"]}
    base_path = paths["Qwen/Qwen3-4B"]
    av_path = paths["anicka/nla-qwen3-4b-universal-av-grpo"]
    ar_path = paths["anicka/nla-qwen3-4b-universal-ar"]
    av_meta = yaml.safe_load((av_path / "nla_meta.yaml").read_text())
    ar_meta = yaml.safe_load((ar_path / "nla_meta.yaml").read_text())
    cfg = json.loads((base_path / "config.json").read_text())
    for meta in (av_meta, ar_meta):
        assert meta["base_model"] == "Qwen/Qwen3-4B"
        assert meta["d_model"] == cfg["hidden_size"]
        assert meta["n_layers"] == cfg["num_hidden_layers"]
    # Verify actual downloaded bytes, not only the remote manifests.
    for m in manifest["models"]:
        for item in m["files"]:
            path = paths[m["repo"]] / item["path"]
            assert path.stat().st_size == item["bytes"]
            with path.open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == item["sha256"], path
    print("All model files verified against pinned SHA-256 manifest", flush=True)
    shapes = {}
    for shard in sorted(base_path.glob("model-*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                shapes[key] = f.get_slice(key).get_shape()
    adapter_checks = []
    for path in (av_path, ar_path):
        params = json.loads((path / "adapter_config.json").read_text())
        count = 0
        with safe_open(path / "adapter_model.safetensors", framework="pt") as f:
            for key in f.keys():
                match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
                assert match, key
                shape = f.get_slice(key).get_shape()
                out_dim, in_dim = shapes[match[1] + ".weight"]
                expected = [params["r"], in_dim] if match[2] == "A" else [out_dim, params["r"]]
                assert shape == expected, (key, shape, expected)
                count += 1
        assert count == cfg["num_hidden_layers"] * len(params["target_modules"]) * 2
        adapter_checks.append({"adapter": path.name, "validated_tensors": count})
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    assert tokenizer.encode(av_meta["tokens"]["injection_char"], add_special_tokens=False) == [av_meta["tokens"]["injection_token_id"]]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).to("cuda").eval()
    model = PeftModel.from_pretrained(base, av_path, adapter_name="av", is_trainable=False)
    model.load_adapter(ar_path, adapter_name="ar", is_trainable=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    backbone = model.get_base_model().model
    layers = [18, 22, 26]
    records = []
    result = {
        "mode": args.mode,
        "av_template": args.av_template,
        "hardware": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "safetensors", "accelerate", "PyYAML"]},
        "cuda_runtime": torch.version.cuda, "manifest": manifest,
        "adapter_checks": adapter_checks, "load_seconds": time.perf_counter() - start,
        "resident_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "resident_parameter_dtypes": sorted(set(str(p.dtype) for p in model.parameters())),
        "cases": [], "interpretations": records,
    }
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Loaded shared base and two adapters", result["resident_parameter_bytes"], flush=True)
    captures = []
    with torch.inference_mode():
        for case_index, (name, prompt) in enumerate(PROMPTS):
            torch.manual_seed(100 + case_index)
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=args.mode == "thinking", return_tensors="pt").to("cuda")
            torch.cuda.synchronize(); start = time.perf_counter()
            with model.disable_adapter():
                seq = inputs if args.mode == "prompt-end" else model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=512,
                    do_sample=True, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0,
                    pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            generated = seq[0, inputs.shape[1]:].tolist()
            close_id = tokenizer.convert_tokens_to_ids("</think>")
            close_at = generated.index(close_id) if close_id in generated else None
            limit = close_at if close_at is not None else len(generated)
            positions = sorted(set(inputs.shape[1] + max(1, int(limit * fraction)) for fraction in [0.35, 0.75]))
            if args.mode == "prompt-end":
                positions = [inputs.shape[1] - 1]
                layers = [22]
            states = {}
            def collect(layer):
                def hook(_module, _args, out):
                    state = out[0] if isinstance(out, tuple) else out
                    for pos in positions:
                        states[layer, pos] = state[0, pos].float().cpu().clone()
                return hook
            handles = [backbone.layers[layer].register_forward_hook(collect(layer)) for layer in layers]
            try:
                with model.disable_adapter():
                    backbone(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False)
            finally:
                for handle in handles: handle.remove()
            # Compare one selected state with a prefix-only pass to check alignment
            # and causal replay. Differences can arise from BF16 kernel shapes.
            prefix_state = []
            pos = positions[0]
            def prefix_hook(_m, _a, out):
                state = out[0] if isinstance(out, tuple) else out
                prefix_state.append(state[0, -1].float().cpu().clone())
            handle = backbone.layers[22].register_forward_hook(prefix_hook)
            try:
                with model.disable_adapter():
                    backbone(input_ids=seq[:, :pos+1], use_cache=False)
            finally: handle.remove()
            causal_error = float((prefix_state[0] - states[22, pos]).norm() / states[22, pos].norm())
            assert causal_error < 0.02, causal_error
            case = {"name": name, "prompt": prompt, "prompt_tokens": inputs.shape[1],
                "generated_tokens": len(generated), "thinking_closed": close_at is not None,
                "finished": bool(generated) and (generated[-1] in model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else generated[-1] == model.generation_config.eos_token_id),
                "text": tokenizer.decode(generated, skip_special_tokens=False),
                "generation_seconds": elapsed, "tokens_per_second": len(generated)/elapsed,
                "prefix_replay_relative_l2": causal_error}
            result["cases"].append(case)
            for pos in positions:
                for layer in layers:
                    captures.append(states[layer, pos])
                    records.append({"case": name, "position": pos, "layer": layer,
                        "consumed_token": tokenizer.decode([int(seq[0, pos])]),
                        "visible_prefix": tokenizer.decode(seq[0, :pos+1], skip_special_tokens=False)})
            print("Target", name, "tokens", len(generated), "thinking_closed", close_at is not None,
                  "tokens/s", round(case["tokens_per_second"], 1), flush=True)
            save()
        model.set_adapter("av")
        for i, (record, act) in enumerate(zip(records, captures)):
            depth = min(av_meta["depth_percentages"], key=lambda p: abs(p-record["layer"]*100/cfg["num_hidden_layers"]))
            content = av_meta["prompt_templates"]["av"].format(depth_pct=depth, injection_char=av_meta["tokens"]["injection_char"])
            template_kwargs = {"enable_thinking": False} if args.av_template == "card" else {}
            ids = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt", **template_kwargs).to("cuda")
            index = (ids[0] == av_meta["tokens"]["injection_token_id"]).nonzero().flatten()
            assert len(index) == 1
            embeds = model.get_input_embeddings()(ids).clone()
            embeds[0, index.item()] = (act / act.norm() * av_meta["extraction"]["injection_scale"]).to(embeds)
            torch.cuda.synchronize(); start = time.perf_counter()
            out = model.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=160, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            raw = tokenizer.decode(out[0], skip_special_tokens=True)
            explanation = raw.split("</explanation>")[0].replace("<explanation>", "").strip()
            record.update(depth_pct=depth, explanation=explanation, av_raw=raw,
                av_tokens=out.shape[1], av_seconds=elapsed, activation_norm=float(act.norm()))
            print("AV", i, record["case"], "L", record["layer"], explanation[:160], flush=True)
            save()
        model.set_adapter("ar")
        reconstructions = []
        for record in records:
            content = ar_meta["prompt_templates"]["ar"].format(depth_pct=record["depth_pct"],
                explanation=record["explanation"], injection_char=ar_meta["tokens"]["injection_char"])
            ids = tokenizer.encode(content, add_special_tokens=False, return_tensors="pt").to("cuda")
            assert ids.shape[1] <= 512 and int(ids[0,-1]) == ar_meta["tokens"]["injection_token_id"]
            state = []
            def readout(_m, _a, out):
                value = out[0] if isinstance(out, tuple) else out
                state.append(value[0, -1].float().cpu().clone())
            handle = backbone.layers[record["layer"]].register_forward_hook(readout)
            try:
                torch.cuda.synchronize(); start=time.perf_counter()
                backbone(input_ids=ids, use_cache=False)
                torch.cuda.synchronize()
                record["ar_seconds"] = time.perf_counter()-start
            finally: handle.remove()
            reconstructions.append(state[0])
        for i, record in enumerate(records):
            other = next(j for j, r in enumerate(records) if r["layer"] == record["layer"] and r["case"] != record["case"])
            record["raw_cosine"] = float(F.cosine_similarity(captures[i], reconstructions[i], dim=0))
            record["swapped_description_raw_cosine"] = float(F.cosine_similarity(captures[i], reconstructions[other], dim=0))
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["free_bytes_at_end"], _ = torch.cuda.mem_get_info()
        result["notes"] = ["One shared base; target, AV, AR scheduled sequentially.",
            "BF16 base and PEFT default adapter precision; no quantization.",
            "Raw cosine is not calibrated confidence; training-corpus layer means are not in this checkpoint bundle.",
            "Four prompts are a feasibility sample, not an explanation-fidelity benchmark."]
        save()
    print("Finished", args.output, "peak GiB", result["peak_allocated_bytes"]/2**30, flush=True)


if __name__ == "__main__":
    main()
