#!/usr/bin/env python3
"""Freeze independent target-capture -> AV fixtures; never import engine code.

Use the unchanged forward_reference operator policy, stock embedding lookup with
one independently normalized replacement, and full causal replay for decoding.
The target fixture is an immutable input. Injection is CPU FP32 norm/divide/scale,
then one cast to model storage. Output limits use forward_reference.tolerance;
normalization uses absolute 2e-5*peak (floor 1e-6), relative 2e-6, fixed here before
engine execution. This script refuses an existing output to prevent replacement.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3
import forward_reference as oracle

ORACLE_SHA = '88e96870ca54e1aa5e1921f668b29b14201b171c3f64a46a66f1b01a957c5d2f'


def evaluate(model, tokens, injection, position, dtype):
    captures, handles = {}, []
    selected = sorted({position, len(tokens)-1})
    blocks = sorted({0, model.config.num_hidden_layers//2, model.config.num_hidden_layers-1})
    def replace(_module, _inputs, output):
        result = output.clone()
        result[0, position] = injection.to(result)
        return result
    handles.append(model.get_input_embeddings().register_forward_hook(replace))
    for block in blocks:
        def before(_module, args, block=block):
            captures[(block, 'block_input')] = oracle.rows(args[0], selected, dtype, model.config.num_hidden_layers)
        def after(_module, _args, result, block=block):
            value = result[0] if isinstance(result, tuple) else result
            captures[(block, 'block_output')] = oracle.rows(value, selected, dtype, model.config.num_hidden_layers)
        handles.extend([model.model.layers[block].register_forward_pre_hook(before),
                        model.model.layers[block].register_forward_hook(after)])
    try:
        ids = torch.tensor([tokens], device=next(model.parameters()).device)
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        logits = oracle.rows(result.logits, [len(tokens)-1], dtype, model.config.num_hidden_layers)
        next_id = int(result.logits[0, -1].argmax())
    finally:
        for handle in handles:
            handle.remove()
    return {'token_ids': tokens.copy(), 'logit_positions': [len(tokens)-1], 'logits': logits,
            'captures': [{'block': block, 'boundary': boundary, **captures[(block, boundary)]}
                         for block in blocks for boundary in ('block_input', 'block_output')]}, next_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-fixture', type=Path, required=True)
    p.add_argument('--av', type=Path)
    p.add_argument('--manifest', type=Path, default=Path(__file__).with_name('models-8b.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('refusing to replace a frozen AV fixture')
    if oracle.digest(Path(oracle.__file__)) != ORACLE_SHA:
        raise ValueError('target oracle hash changed')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    target = json.loads(a.target_fixture.read_text())
    actual = a.av is not None
    root = a.av if actual else a.target_fixture.parent
    dtype = target['dtype']
    capture_block = 24 if actual else 0
    target_position = target['generation']['prompt_tokens']-1
    capture = next(c for c in target['generation']['reference']['captures']
                   if c['block'] == capture_block and c['boundary'] == 'block_output')
    offset = capture['positions'].index(target_position)*capture['width']
    h = torch.tensor(capture['values'][offset:offset+capture['width']], dtype=torch.float32)
    if actual:
        entry = next(m for m in json.loads(a.manifest.read_text())['models'] if m['repo']=='marco-bazzani/Qwen3-8B-nla')
        artifacts = []
        for f in entry['files']:
            if f['path'].startswith('ar/'):
                continue
            path = root/f['path']
            assert path.stat().st_size == f['bytes'] and oracle.digest(path) == f['sha256']
            artifacts.append({'relative_path':f['path'],'byte_size':f['bytes'],'sha256':f['sha256']})
        meta = yaml.safe_load((root/'nla_meta.yaml').read_text())
        tok = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        text = meta['prompt_templates']['actor'].format(injection_char=meta['tokens']['injection_char'])
        formatted = tok.apply_chat_template([{'role':'user','content':text}], tokenize=False, add_generation_prompt=True)
        prompt = tok.encode(formatted, add_special_tokens=False)
        position = prompt.index(meta['tokens']['injection_token_id'])
        assert prompt.count(meta['tokens']['injection_token_id']) == 1
        assert prompt[position-1] == meta['tokens']['injection_left_neighbor_id']
        assert prompt[position+1] == meta['tokens']['injection_right_neighbor_id']
        scale = float(meta['extraction']['injection_scale'])
        repo, rev = entry['repo'], entry['revision']
        stops = json.loads((root/'generation_config.json').read_text())['eos_token_id']
    else:
        artifacts = target['artifacts']
        prompt, position, scale, formatted = [3, 7, 11, 2, 5], 3, 3.0, ''
        repo, rev, stops = target['repository'], target['revision'], [0]
    n = h.norm()
    assert torch.isfinite(h).all() and torch.isfinite(n) and n > 0
    normalized = h/n*scale
    injection = normalized.to(getattr(torch, dtype))
    model = AutoModelForCausalLM.from_pretrained(root, torch_dtype=getattr(torch,dtype),
        attn_implementation='eager', local_files_only=True, trust_remote_code=False,
        device_map='cuda').eval().requires_grad_(False)
    print('Loaded AV; generating independent reference', flush=True)
    original_attention, original_linear = modeling_qwen3.eager_attention_forward, torch.nn.Linear.forward
    modeling_qwen3.eager_attention_forward = oracle.invariant_attention
    torch.nn.Linear.forward = oracle.invariant_linear
    try:
        tokens, steps = prompt.copy(), []
        for _ in range(4):
            result, next_id = evaluate(model,tokens,injection,position,dtype)
            steps.append(result)
            tokens.append(next_id)
            if next_id in stops:
                break
        final, _ = evaluate(model,tokens,injection,position,dtype)
    finally:
        modeling_qwen3.eager_attention_forward, torch.nn.Linear.forward = original_attention, original_linear
    out = {'schema_version':1,'target_fixture_sha256':oracle.digest(a.target_fixture),
           'generator_sha256':oracle.digest(Path(__file__)), 'oracle_sha256':ORACLE_SHA,
           'environment':oracle.environment('cuda'), 'operator_policy':'unchanged forward_reference.py',
           'torch_qwen3_sha256':oracle.digest(Path(inspect.getsourcefile(modeling_qwen3))),
           'repository':repo,'revision':rev,'root':str(root.resolve()),'dtype':dtype,
           'artifacts':artifacts,'shards':sorted(f['relative_path'] for f in artifacts if f['relative_path'].endswith('.safetensors')),
           'index':'model.safetensors.index.json' if actual else None,
           'capture_block':capture_block,'capture_position':target_position,'capture_values':h.tolist(),
           'norm':float(n),'scale':scale,'normalized_fp32':normalized.tolist(),
           'normalization_tolerance':{'max_absolute_error':max(1e-6,float(normalized.abs().max())*2e-5),'max_relative_l2':2e-6},
           'injection_values':injection.float().tolist(),
           'injection_sha256':hashlib.sha256(injection.view(torch.uint8).numpy().tobytes()).hexdigest(),
           'formatted_prompt':formatted,'prompt_ids':prompt,'injection_position':position,
           'stop_ids':stops,'max_new_tokens':4,'steps':steps,'reference':final,
           'expected_generated_ids':tokens[len(prompt):]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(out,f,allow_nan=False);f.write('\n')
    print('Frozen',a.output,oracle.digest(a.output),flush=True)


if __name__ == '__main__':
    main()
