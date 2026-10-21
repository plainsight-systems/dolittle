#!/usr/bin/env python3
"""Freeze AR and single-position steering expectations independently of Dolittle.

AR follows kitft/nla-inference NLACritic at revision
38b802a33d1d317f21b6825a9116f388c2141f86: truncated backbone, no final
normalization, last-token residual through the separately trained value head.
Use the unchanged forward_reference.py numerical operator policy and limits.
Direction/reference mutation use ordinary FP32 PyTorch operations; their bounds
are fixed here before engine comparison. Never overwrite a frozen fixture.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3
import forward_reference as oracle

ORACLE_SHA = '88e96870ca54e1aa5e1921f668b29b14201b171c3f64a46a66f1b01a957c5d2f'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-fixture', type=Path, required=True)
    p.add_argument('--ar', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('refusing to replace frozen steering fixture')
    assert oracle.digest(Path(oracle.__file__)) == ORACLE_SHA
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    target = json.loads(a.target_fixture.read_text())
    dtype = target['dtype']; storage = getattr(torch, dtype)
    root = a.target_fixture.parent
    actual = a.ar is not None
    config = json.loads((root/'config.json').read_text())
    d = config['hidden_size']
    if actual:
        arroot = a.ar.parent
        entry = next(m for m in json.loads(Path(__file__).with_name('models-8b.json').read_text())['models'] if m['repo']=='marco-bazzani/Qwen3-8B-nla')
        artifacts = [{'relative_path':f['path'],'byte_size':f['bytes'],'sha256':f['sha256']} for f in entry['files'] if f['path'].startswith('ar/')]
        for f in artifacts:
            path = arroot/f['relative_path']; assert path.stat().st_size==f['byte_size'] and oracle.digest(path)==f['sha256']
        repo, rev = entry['repo'], entry['revision']
        modelroot = a.ar; config_name = 'ar/config.json'
        tokenizer = AutoTokenizer.from_pretrained(modelroot, local_files_only=True, trust_remote_code=False)
        descriptions = ['A rabbit is looking at a carrot.', 'A mouse is looking at a carrot.']
        prompts = [tokenizer.encode('Summary of the following text: <text>'+s+'</text> <summary>', add_special_tokens=True) for s in descriptions]
        assert all(ids[-5:]==[1318,29,366,1708,29] for ids in prompts)
    else:
        arroot = a.output.parent/'ar'; arroot.mkdir(parents=True,exist_ok=False)
        weights = load_file(root/'model.safetensors')
        weights = {k:v for k,v in weights.items() if k!='model.norm.weight' and k!='lm_head.weight'}
        torch.manual_seed(20261020)
        head = torch.randn(d,d,dtype=torch.float32).mul(0.05).to(storage)
        save_file(weights, arroot/'model.safetensors'); save_file({'weight':head}, arroot/'value_head.safetensors')
        (arroot/'config.json').write_text(json.dumps(config))
        artifacts = [{'relative_path':f.name,'byte_size':f.stat().st_size,'sha256':oracle.digest(f)} for f in sorted(arroot.iterdir())]
        repo,rev='fixture/qwen3-ar','20261020'; modelroot=arroot;config_name='config.json'
        descriptions=['original','edited'];prompts=[[3,7,11,2,5],[3,7,9,2,5]]
    old_attention,old_linear=modeling_qwen3.eager_attention_forward,torch.nn.Linear.forward
    modeling_qwen3.eager_attention_forward=oracle.invariant_attention;torch.nn.Linear.forward=oracle.invariant_linear
    try:
        ar=AutoModelForCausalLM.from_pretrained(modelroot,torch_dtype=storage,attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval().requires_grad_(False)
        ar.model.norm=torch.nn.Identity();ar.lm_head=torch.nn.Identity();ar=ar.to('cuda')
        head=torch.nn.Linear(d,d,bias=False,dtype=storage)
        head.load_state_dict(load_file(modelroot/'value_head.safetensors'));head=head.to('cuda').eval().requires_grad_(False)
        vectors=[]
        with torch.inference_mode():
            for ids in prompts:
                hidden=ar.model(torch.tensor([ids],device='cuda'),use_cache=False).last_hidden_state[0,-1]
                vectors.append(head(hidden).detach().cpu())
        ar_blocks=ar.config.num_hidden_layers
        del ar,head;gc.collect();torch.cuda.empty_cache()
        diff=vectors[1].float()-vectors[0].float();direction=diff/diff.norm()
        vector_rows=[{'values':v.float().tolist(),'width':d,'tolerance':oracle.tolerance(v.float().tolist(),dtype,ar_blocks)} for v in vectors]
        armeta={'repository':repo,'revision':rev,'root':str(arroot.resolve()),'artifacts':artifacts,'shards':[f['relative_path'] for f in artifacts if f['relative_path'].endswith('.safetensors')],'index':None,'config':config_name,'prompts':prompts,'descriptions':descriptions,'vectors':vector_rows}
        model=AutoModelForCausalLM.from_pretrained(root,torch_dtype=storage,attn_implementation='eager',local_files_only=True,trust_remote_code=False,device_map='cuda').eval().requires_grad_(False)
        block=24 if actual else 0
        original=target['generation']['reference']['token_ids']
        prompt_size=target['generation']['prompt_tokens']
        cases=[]
        # Prompt position and generated position exercise both replay boundaries.
        for position in [prompt_size-1,prompt_size]:
            prefix=original[:max(prompt_size,position+1)]
            for strength in [0.0,0.5]:
                tokens=prefix.copy(); captured={}
                def mutate(_module,_args,result):
                    output=result[0] if isinstance(result,tuple) else result
                    h=output[0,position].float()
                    changed=(h+strength*h.norm()*direction.to(h.device)).to(storage) if strength else output[0,position]
                    captured['before']=output[0,position].detach().cpu();captured['after']=changed.detach().cpu()
                    output=output.clone();output[0,position]=changed
                    return (output,*result[1:]) if isinstance(result,tuple) else output
                handle=model.model.layers[block].register_forward_hook(mutate)
                try:
                    with torch.inference_mode():
                        for _ in range(3):
                            logits=model(torch.tensor([tokens],device='cuda'),use_cache=False).logits[0,-1]
                            tokens.append(int(logits.argmax()))
                finally:handle.remove()
                cases.append({'position':position,'prefix':prefix,'strength':strength,'generated':tokens[len(prefix):],
                              'before':captured['before'].float().tolist(),'after':captured['after'].float().tolist(),
                              'tolerance':oracle.tolerance(captured['after'].float().tolist(),dtype,config['num_hidden_layers'])})
    finally:
        modeling_qwen3.eager_attention_forward=old_attention;torch.nn.Linear.forward=old_linear
    out={'schema':1,'target_fixture_sha256':oracle.digest(a.target_fixture),'generator_sha256':oracle.digest(Path(__file__)),'oracle_sha256':ORACLE_SHA,'environment':oracle.environment('cuda'),'dtype':dtype,'ar':armeta,'direction':direction.tolist(),
         'direction_tolerance':{'max_absolute_error':max(1e-6,float(direction.abs().max())*2e-5),'max_relative_l2':2e-6},'block':block,'cases':cases}
    with a.output.open('x') as f:json.dump(out,f,allow_nan=False);f.write('\n')
    print('Frozen steering fixture',a.output,oracle.digest(a.output),flush=True)

if __name__=='__main__':main()
