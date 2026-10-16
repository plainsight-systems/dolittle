#!/usr/bin/env python3
"""Run the published Qwen3-8B NLA against the existing four-case 4B workload.

Contract: a verified models-8b.json bundle in --root and a BF16 CUDA GPU.
This is reference tooling, not Dolittle engine code or a fidelity benchmark.
Use the same seeds, target sampling, replay capture fractions, and output caps
as reference_probe.py. Each target has its own generated trajectory; vectors
and reconstruction scores cannot be compared directly between models.

Design: target, full AV, and truncated AR occupy the GPU in separate phases.
Optimization: retain only selected CPU residuals; release weights and caches
between phases. Four prompt-end controls accompany eight thinking captures.
The AV uses its checkpoint's default chat template. The AR bypasses final RMS
normalization and applies its separate value head, matching kitft/nla-inference
38b802a33d1d317f21b6825a9116f388c2141f86. No downloaded code is executed.
"""

import argparse
import gc
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
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from reference_probe import PROMPTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    manifest = json.loads((args.root / 'models.json').read_text())
    paths = {m['repo']: args.root / 'models' / m['repo'].replace('/', '--')
             for m in manifest['models']}
    for m in manifest['models']:
        for item in m['files']:
            path = paths[m['repo']] / item['path']
            assert path.stat().st_size == item['bytes'], path
            with path.open('rb') as source:
                assert hashlib.file_digest(source, 'sha256').hexdigest() == item['sha256'], path
    print('All pinned files verified', flush=True)
    base_path = paths['Qwen/Qwen3-8B']
    av_path = paths['marco-bazzani/Qwen3-8B-nla']
    ar_path = av_path / 'ar'
    av_meta = yaml.safe_load((av_path / 'nla_meta.yaml').read_text())
    ar_meta = yaml.safe_load((ar_path / 'nla_meta.yaml').read_text())
    assert av_meta['base_checkpoint'] == 'Qwen/Qwen3-8B'
    assert av_meta['d_model'] == ar_meta['d_model'] == 4096
    result = {
        'manifest': manifest, 'hardware': torch.cuda.get_device_name(),
        'compute_capability': torch.cuda.get_device_capability(),
        'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
        'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                  '--format=csv,noheader'], text=True).strip(),
        'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
        'packages': {p: importlib.metadata.version(p) for p in
                     ['torch', 'transformers', 'peft', 'safetensors', 'accelerate', 'PyYAML']},
        'cases': [], 'interpretations': [], 'phases': {},
        'notes': ['BF16, no quantization; one model loaded at a time.',
                  'Four prompts, eight thinking captures and four prompt-end controls.',
                  'Raw cosine is diagnostic, not semantic confidence or cross-model accuracy.',
                  'Causal replay, not capture inside cached decoding.'],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    def phase_start():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return time.perf_counter()

    def phase_end(name, model, started):
        torch.cuda.synchronize()
        result['phases'][name] = {
            'seconds': time.perf_counter() - started,
            'resident_parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        }
        save()

    def load(path):
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True,
            trust_remote_code=False, device_map='cuda').eval().requires_grad_(False)

    def capture(model, seq, positions):
        states = {}
        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            for pos in positions:
                states[pos] = h[0, pos].float().cpu().clone()
        handle = model.model.layers[24].register_forward_hook(hook)
        try:
            model.model(input_ids=seq, attention_mask=torch.ones_like(seq), use_cache=False)
        finally:
            handle.remove()
        for state in states.values():
            assert state.shape == (4096,) and torch.isfinite(state).all()
        return states

    captures = []
    with torch.inference_mode():
        started = phase_start()
        model = load(base_path)
        torch.cuda.synchronize()
        target_load_seconds = time.perf_counter() - started
        tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
        print('Target loaded', flush=True)
        for case_index, (name, prompt) in enumerate(PROMPTS):
            torch.manual_seed(100 + case_index)
            ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=True, return_tensors='pt').to('cuda')
            torch.cuda.synchronize(); start = time.perf_counter()
            seq = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=512,
                do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            generated = seq[0, ids.shape[1]:].tolist()
            close_id = tok.convert_tokens_to_ids('</think>')
            close_at = generated.index(close_id) if close_id in generated else None
            limit = close_at if close_at is not None else len(generated)
            assert limit > 1
            positions = sorted(set(ids.shape[1] + max(1, int(limit * f)) for f in [0.35, 0.75]))
            assert all(p < ids.shape[1] + limit for p in positions)
            states = capture(model, seq, positions)
            pos = positions[0]
            prefix_state = capture(model, seq[:, :pos+1], [pos])[pos]
            error = float((prefix_state - states[pos]).norm() / states[pos].norm())
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            result['cases'].append({'name': name, 'prompt': prompt, 'prompt_tokens': ids.shape[1],
                'generated_tokens': len(generated), 'thinking_closed': close_at is not None,
                'finished': generated[-1] in eos, 'text': tok.decode(generated, skip_special_tokens=False),
                'generation_seconds': elapsed, 'tokens_per_second': len(generated) / elapsed,
                'prefix_replay_relative_l2': error})
            for p in positions:
                captures.append(states[p])
                result['interpretations'].append({'case': name, 'mode': 'thinking', 'layer': 24,
                    'position': p, 'consumed_token': tok.decode([int(seq[0, p])]),
                    'visible_prefix': tok.decode(seq[0, :p+1], skip_special_tokens=False)})
            control = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to('cuda')
            p = control.shape[1] - 1
            captures.append(capture(model, control, [p])[p])
            result['interpretations'].append({'case': name, 'mode': 'prompt-end', 'layer': 24,
                'position': p, 'consumed_token': tok.decode([int(control[0, p])]),
                'visible_prefix': tok.decode(control[0], skip_special_tokens=False)})
            print('Target', name, len(generated), 'tokens,', round(len(generated)/elapsed, 2),
                  'tokens/s, replay L2', round(error, 5), flush=True)
            save()
        phase_end('target', model, started)
        result['phases']['target']['load_seconds'] = target_load_seconds
        del model, seq, ids, control, states, prefix_state

        started = phase_start()
        model = load(av_path)
        torch.cuda.synchronize()
        av_load_seconds = time.perf_counter() - started
        tok = AutoTokenizer.from_pretrained(av_path, local_files_only=True, trust_remote_code=False)
        print('AV loaded', flush=True)
        content = av_meta['prompt_templates']['actor'].format(
            injection_char=av_meta['tokens']['injection_char'])
        ids = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True,
            add_generation_prompt=True, return_tensors='pt').to('cuda')
        index = (ids[0] == av_meta['tokens']['injection_token_id']).nonzero().flatten()
        assert len(index) == 1
        pos = index.item()
        assert int(ids[0, pos-1]) == av_meta['tokens']['injection_left_neighbor_id']
        assert int(ids[0, pos+1]) == av_meta['tokens']['injection_right_neighbor_id']
        result['av_formatted_prompt'] = tok.decode(ids[0], skip_special_tokens=False)
        for i, (record, act) in enumerate(zip(result['interpretations'], captures)):
            embeds = model.get_input_embeddings()(ids).clone()
            embeds[0, pos] = (act / act.norm() * av_meta['extraction']['injection_scale']).to(embeds)
            torch.cuda.synchronize(); start = time.perf_counter()
            out = model.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                do_sample=False, max_new_tokens=160, pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            raw = tok.decode(out[0], skip_special_tokens=True)
            match = re.search(r'<explanation>\s*(.*?)\s*</explanation>', raw, flags=re.S)
            explanation = match.group(1).strip() if match else raw.strip()
            record.update(explanation=explanation, av_raw=raw, explanation_tags_complete=bool(match),
                av_tokens=out.shape[1], av_seconds=elapsed, activation_norm=float(act.norm()),
                av_hit_token_cap=out.shape[1] == 160)
            print('AV', i, record['case'], record['mode'], explanation[:180], flush=True)
            save()
        phase_end('av', model, started)
        result['phases']['av']['load_seconds'] = av_load_seconds
        del model, embeds, out, ids

        started = phase_start()
        # Load CPU first so unused lm_head never consumes GPU memory.
        model = AutoModelForCausalLM.from_pretrained(ar_path, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True, trust_remote_code=False)
        assert model.config.num_hidden_layers == 28
        model.lm_head = torch.nn.Identity()
        model.model.norm = torch.nn.Identity()
        model = model.to('cuda').eval().requires_grad_(False)
        head = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.bfloat16)
        head.load_state_dict(load_file(str(ar_path / 'value_head.safetensors')))
        head = head.to('cuda').eval().requires_grad_(False)
        torch.cuda.synchronize()
        ar_load_seconds = time.perf_counter() - started
        tok = AutoTokenizer.from_pretrained(ar_path, local_files_only=True, trust_remote_code=False)
        predictions = []
        for record in result['interpretations']:
            prompt = ar_meta['prompt_templates']['critic'].format(explanation=record['explanation'])
            ids = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'].to('cuda')
            suffix = ar_meta['tokens']['critic_suffix_ids']
            assert ids[0, -len(suffix):].tolist() == suffix
            torch.cuda.synchronize(); start = time.perf_counter()
            hidden = model.model(ids, use_cache=False).last_hidden_state[0, -1]
            pred = head(hidden).float().cpu()
            torch.cuda.synchronize()
            record['ar_seconds'] = time.perf_counter() - start
            assert torch.isfinite(pred).all()
            predictions.append(pred)
        for i, record in enumerate(result['interpretations']):
            other = next(j for j, r in enumerate(result['interpretations'])
                         if r['mode'] == record['mode'] and r['case'] != record['case'])
            record['raw_cosine'] = float(F.cosine_similarity(captures[i], predictions[i], dim=0))
            record['swapped_description_raw_cosine'] = float(F.cosine_similarity(captures[i], predictions[other], dim=0))
        phase_end('ar', model, started)
        result['phases']['ar']['load_seconds'] = ar_load_seconds
        result['phases']['ar']['value_head_bytes'] = sum(p.numel()*p.element_size() for p in head.parameters())
        save()
    print('Finished', args.output, flush=True)


if __name__ == '__main__':
    main()
