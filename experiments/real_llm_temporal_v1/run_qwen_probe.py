#!/usr/bin/env python3
"""L01 real-LLM temporal KV intervention pilot.

Pinned for transformers==4.57.1 with Qwen2Attention, attn_implementation='eager'.
Logical compression via 4D additive attention masks implements positive-mass
(2,0) pair thinning without physically freeing KV storage.

Full-checkpoint integration requires the cached Qwen2.5 weights and CUDA.
--hf-smoke-tiny checks the pinned HF interfaces with random small weights and
can run on CPU; --self-test runs the CPU schedule, mask, and summary checks.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import random
import sys
import time
from typing import Any


def stable_seed(*items: Any) -> int:
    blob=json.dumps(items, ensure_ascii=False, separators=(',',':')).encode('utf8')
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], 'big') % (2**63 - 1)


def event_count(horizon: int) -> int:
    """Pair (y1,y2) first affects predicted y4, etc. T=130 -> 64 events."""
    return max(0,(horizon - 2)//2)


def signs_for(policy: str, phase: int, n_events: int, random_seed: int = 0) -> list[int]:
    if policy not in ('block','alternating','balanced_random') or phase not in (-1,1):
        raise ValueError('unknown policy/phase')
    if policy == 'alternating':
        return [phase * (1 if i % 2 == 0 else -1) for i in range(n_events)]
    if policy == 'balanced_random':
        base=[1]*(n_events//2)+[-1]*(n_events-n_events//2)
        random.Random(random_seed).shuffle(base)
        return [phase*x for x in base]
    # For even n_events, precisely half + and half - in either phase.
    return [phase * (1 if i < n_events//2 else -1) for i in range(n_events)]


def apply_pair_bias(bias: 'torch.Tensor', prompt_len: int, event_index: int, sign: int) -> tuple[int,int]:
    """Mutate additive scores to equivalent positive weights (2,0)."""
    import torch
    i=prompt_len+2*event_index
    j=i+1
    if i<0 or j>=bias.numel() or sign not in (-1,1):
        raise ValueError('pair is not present in cache')
    keep,drop=(i,j) if sign==1 else (j,i)
    bias[keep]=math.log(2.)
    bias[drop]=torch.finfo(bias.dtype).min
    return i,j


def make_four_d_mask(bias: 'torch.Tensor', cache_len: int, dtype: 'torch.dtype' = None) -> 'torch.Tensor':
    import torch
    if bias.numel()!=cache_len:
        raise AssertionError(f'expected {cache_len} mask entries, got {bias.numel()}')
    # Qwen eager attention adds the mask to the score tensor before softmax.
    # Keep log(2) in FP32 even with BF16 queries; casting it first rounds the
    # intervention weight and changes the intended positive mass of 2.
    del dtype  # retained as a compatibility argument; query dtype is irrelevant here
    return bias.to(dtype=torch.float32).view(1,1,1,-1)


def eos_ids_for(tokenizer, model) -> set[int]:
    """Collect every configured EOS token ID without changing the fixed horizon."""
    values=[getattr(tokenizer,'eos_token_id',None),
            getattr(getattr(model,'generation_config',None),'eos_token_id',None)]
    ids=set()
    for value in values:
        if value is None:
            continue
        if isinstance(value,(list,tuple,set)):
            ids.update(int(x) for x in value if x is not None)
        else:
            ids.add(int(value))
    return ids


def public_model_reference(model_path: str) -> str:
    """Return a checkpoint label while keeping absolute machine paths private."""
    posix_path=Path(model_path)
    windows_path=PureWindowsPath(model_path)
    if posix_path.is_absolute():
        return posix_path.name
    if windows_path.is_absolute():
        return windows_path.name
    return model_path


def kl_full_softmax(q_logits: 'torch.Tensor', p_logits: 'torch.Tensor') -> float:
    import torch
    import torch.nn.functional as F
    q=q_logits.reshape(-1).float();p=p_logits.reshape(-1).float()
    q_log=F.log_softmax(q,dim=-1)
    p_log=F.log_softmax(p,dim=-1)
    value=float((q_log.exp()*(q_log-p_log)).sum().item())
    if not math.isfinite(value) or value < -1e-4:
        raise FloatingPointError(f'invalid KL: {value}')
    return max(0.,value)


def sample_inverse_cdf(logits: 'torch.Tensor', u: float) -> int:
    import torch
    probs=torch.softmax(logits.reshape(-1).float(),dim=-1)
    idx=torch.searchsorted(probs.cumsum(0),torch.tensor(u,device=probs.device,dtype=probs.dtype)).item()
    return min(int(idx),probs.numel()-1)


def cpu_math_self_test() -> None:
    import torch
    for horizon in (8,32,130):
        n=event_count(horizon)
        for pol in ('block','alternating','balanced_random'):
            sp=signs_for(pol,1,n); sm=signs_for(pol,-1,n)
            assert all(a==-b for a,b in zip(sp,sm))
            if n%2==0:
                assert sum(sp)==sum(sm)==0
    # Compare mathematical weighted attention against additive-mask simulation.
    for seed in range(30):
        rng=random.Random(seed)
        x=torch.tensor([rng.uniform(-3,3) for _ in range(7)],dtype=torch.float64)
        v=torch.tensor([rng.uniform(-2,2) for _ in range(7)],dtype=torch.float64)
        for sign in (-1,1):
            bias=torch.zeros(7,dtype=torch.float64)
            apply_pair_bias(bias,3,0,sign)
            kept=3 if sign==1 else 4
            indices=[i for i in range(7) if i not in (3,4)]+[kept]
            weight=[1.]*5+[2.]
            expected=(torch.softmax(x[indices]+torch.tensor(weight,dtype=torch.float64).log(),0)*v[indices]).sum()
            got=(torch.softmax(x+bias,0)*v).sum()
            assert torch.allclose(got,expected,atol=1e-12,rtol=0), (got,expected)
        assert kl_full_softmax(x,x)==0.
    # A BF16 query must still receive an FP32 additive bias so log(2) is not
    # rounded before the eager-attention score addition.
    log_two=math.log(2.)
    rounded_bf16=float(torch.tensor(log_two,dtype=torch.bfloat16).float().item())
    assert rounded_bf16!=log_two
    fp32_mask=make_four_d_mask(torch.tensor([0.,log_two],dtype=torch.float32),2,torch.bfloat16)
    assert fp32_mask.dtype==torch.float32
    assert abs(float(fp32_mask[0,0,0,1].item())-log_two)<1e-7
    print('CPU SELF-TEST PASS: time schedules, antiphases, (2,0) attention equivalence, identity KL.')


class AttentionLocalProbe:
    """Monkeypatch Qwen2 eager attention *only* when evaluating this branch.

    At the first token affected by a new event, compares outputs of the same
    attention query on the same accumulated Q cache, with vs. without ONLY that
    latest edit; all earlier edits are unchanged. This is a direct local
    attention-response diagnostic, not a counterfactual full-model KL.
    """
    def __init__(self):
        self.latest_pair=None
        self.errors=[]
        self._old_fn=None
        self._mod=None

    def __enter__(self):
        import transformers.models.qwen2.modeling_qwen2 as mod
        self._mod=mod
        self._old_fn=mod.eager_attention_forward
        orig=self._old_fn
        owner=self
        def wrapper(module,query,key,value,attention_mask,scaling,dropout=0.0,**kwargs):
            out,weights=orig(module,query,key,value,attention_mask,scaling,dropout=dropout,**kwargs)
            if owner.latest_pair is not None:
                if attention_mask is None or attention_mask.ndim!=4:
                    raise RuntimeError('Qwen2 no longer forwards the custom 4D mask')
                p0,p1=owner.latest_pair
                if key.shape[-2]<=p1 or attention_mask.shape[-1]<=p1:
                    raise RuntimeError('probe pair outside cache or mask')
                baseline=attention_mask.clone()
                baseline[:,:,:,p0]=0.
                baseline[:,:,:,p1]=0.
                old_out,_=orig(module,query,key,value,baseline,scaling,dropout=dropout,**kwargs)
                # Output is pre-o_proj; take full hidden-vector L2^2 per layer.
                diff=(out-old_out).float()
                if diff.ndim!=4:
                    raise RuntimeError(f'unexpected Qwen2 eager output layout: {tuple(diff.shape)}')
                # Return one full-vector squared L2 error per [batch, query],
                # then average those values. Heads are part of the vector norm.
                owner.errors.append((int(module.layer_idx),float(diff.square().sum(dim=(-1,-2)).mean().item())))
            return out,weights
        mod.eager_attention_forward=wrapper
        return self

    def __exit__(self,exc_type,exc,tb):
        if self._mod is not None and self._old_fn is not None:
            self._mod.eager_attention_forward=self._old_fn
        self.latest_pair=None
        return False


def load_prompt_list(path: Path) -> list[dict[str,str]]:
    data=json.loads(path.read_text(encoding='utf8'))
    if not isinstance(data,list) or not data:
        raise ValueError('prompts.json must be a nonempty list')
    out=[]
    for p in data:
        if not isinstance(p,dict) or not all(k in p for k in ('id','category','user')):
            raise ValueError('prompt requires id, category, user')
        if not isinstance(p['user'],str) or len(p['user'])<35:
            raise ValueError('prompt must be a substantive user message')
        out.append(p)
    if len(set(p['id'] for p in out)) != len(out):
        raise ValueError('prompt ids must be unique')
    return out


def encoded_prompt(tokenizer,prompt:dict[str,str],device) -> 'torch.Tensor':
    import torch
    system='You are a precise assistant. Answer the request directly and in detail. Do not stop until the requested explanation is complete.'
    ids=tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':prompt['user']}],tokenize=True,add_generation_prompt=True)
    if not isinstance(ids,list):
        raise TypeError('unexpected tokenizer chat template output')
    return torch.tensor([ids],dtype=torch.long,device=device)


def run_policy(model,tokenizer,input_ids,*,policy,phase,horizon,schedule_seed,trajectory_seed,probe:AttentionLocalProbe):
    import torch
    prompt_len=int(input_ids.shape[1]);device=input_ids.device; dtype=next(model.parameters()).dtype
    events=0 if policy=='identity' else event_count(horizon)
    sign_schedule=[] if policy=='identity' else signs_for(policy,phase,events,random_seed=schedule_seed)
    # Independent exact prefills avoid implementation-dependent cache cloning.
    with torch.inference_mode():
        p_out=model(input_ids=input_ids,use_cache=True,return_dict=True)
        q_out=model(input_ids=input_ids,use_cache=True,return_dict=True)
        p_cache,q_cache=p_out.past_key_values,q_out.past_key_values
        p_logits=p_out.logits[:,-1];q_logits=q_out.logits[:,-1]
        if float((q_logits-p_logits).float().abs().max())>1e-3:
            raise AssertionError('full-cache prefill identity failed')
        q_bias=torch.zeros(prompt_len,device=device,dtype=torch.float32)
        p_bias=torch.zeros(prompt_len,device=device,dtype=torch.float32)
        prng=random.Random(trajectory_seed)
        uniforms=[prng.random() for _ in range(horizon)]
        conditional_kls=[];sampled=[];event_local=[]
        eos_ids=eos_ids_for(tokenizer,model)
        for t in range(horizon):
            if t>0:
                # Next input is previously generated y_t (1-based): index prompt_len+t-1.
                position=prompt_len+t-1
                pair=None
                if policy!='identity' and t>=3 and t%2==1:
                    idx=(t-3)//2
                    assert idx<events
                    q_bias=torch.cat((q_bias,torch.zeros(1,device=device,dtype=q_bias.dtype)))
                    p_bias=torch.cat((p_bias,torch.zeros(1,device=device,dtype=p_bias.dtype)))
                    pair=apply_pair_bias(q_bias,prompt_len,idx,sign_schedule[idx])
                else:
                    q_bias=torch.cat((q_bias,torch.zeros(1,device=device,dtype=q_bias.dtype)))
                    p_bias=torch.cat((p_bias,torch.zeros(1,device=device,dtype=p_bias.dtype)))
                if q_cache.get_seq_length()!=position or p_cache.get_seq_length()!=position:
                    raise AssertionError(f'cache position mismatch: t={t}, expected={position}')
                tok=torch.tensor([[sampled[-1]]],device=device,dtype=torch.long)
                pos=torch.tensor([[position]],device=device,dtype=torch.long)
                cp=pos.squeeze(0)
                kwargs=dict(input_ids=tok,position_ids=pos,cache_position=cp,use_cache=True,return_dict=True)
                p_mask=make_four_d_mask(p_bias,position+1,dtype)
                q_mask=make_four_d_mask(q_bias,position+1,dtype)
                probe.latest_pair=None
                p_out=model(**kwargs,past_key_values=p_cache,attention_mask=p_mask)
                p_cache=p_out.past_key_values
                p_logits=p_out.logits[:,-1]
                probe.errors=[];probe.latest_pair=pair
                q_out=model(**kwargs,past_key_values=q_cache,attention_mask=q_mask)
                q_cache=q_out.past_key_values
                q_logits=q_out.logits[:,-1]
                probe.latest_pair=None
                if pair is not None:
                    if len(probe.errors)!=int(model.config.num_hidden_layers):
                        raise AssertionError('expected exactly one attention probe per Qwen2 decoder layer')
                    local_mse=sum(v for _,v in probe.errors)/len(probe.errors)
                    event_local.append({'event_index':len(event_local),'local_attention_error_sq':local_mse,
                                        'layer_errors':probe.errors})
                if pair is None and not event_local and float((q_logits-p_logits).float().abs().max())>0.04:
                    raise AssertionError('P/Q logits diverged before first compression event')
            # Both conditional dists at this step use the *same* Q-sampled history.
            conditional_kls.append(kl_full_softmax(q_logits,p_logits))
            sampled.append(sample_inverse_cdf(q_logits,uniforms[t]))
        if len(event_local)!=events:
            raise AssertionError('wrong event count')
        if q_cache.get_seq_length()!=prompt_len+horizon-1:
            raise AssertionError('unexpected last logical cache length')
        first_eos=next((i+1 for i,x in enumerate(sampled) if x in eos_ids),None)
        return {
          'policy':policy,'phase':phase,'horizon':horizon,'events':events,
          'sign_schedule':sign_schedule,
          'schedule_seed':int(schedule_seed),'trajectory_seed':int(trajectory_seed),
          'full_record_count_final':int(len(q_bias)),
          'logical_record_count_final':int(len(q_bias)-events),
          'local_metric':'full_head_vector_l2_squared_pre_o_proj',
          'full_kl_q_to_p':float(sum(conditional_kls)),
          'conditional_kl_by_generated_position':conditional_kls,
          'local_attention_error_sq_mean':float(sum(e['local_attention_error_sq'] for e in event_local)/max(1,events)),
          'local_attention_error_sq_q90':float(sorted(e['local_attention_error_sq'] for e in event_local)[math.floor(.9*(events-1))]) if events else 0.,
          'event_local':event_local,
          'kl_milestones':{str(j):float(sum(conditional_kls[:j])) for j in (16,32,64,96,horizon) if j<=horizon},
          'sampled_ids':sampled,
          'eos_token_ids':sorted(eos_ids),
          'eos_count':sum(int(x in eos_ids) for x in sampled),
          'first_eos_position':first_eos,
          'first_eos_position_convention':'1-based generated-token position; null means no EOS',
          'eos_handling':'fixed_horizon_continues_after_first_eos',
          'text':tokenizer.decode(sampled,skip_special_tokens=False)[:3500],
        }


def hf_smoke_tiny(device: str = 'cpu', dtype: str = 'float32') -> None:
    """HF integration test: native/4D parity, probe neutrality, and both (2,0) orientations.

    Requires transformers==4.57.1, torch; uses RANDOM small Qwen2 weights;
    does not load/claim to test an LLM checkpoint.
    """
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM
    if transformers.__version__!='4.57.1':
        raise RuntimeError('HF integration test is pinned to transformers==4.57.1')
    if dtype not in ('float32','bfloat16'):
        raise ValueError('tiny smoke dtype must be float32 or bfloat16')
    device=torch.device(device)
    torch_dtype=getattr(torch,dtype)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for tiny smoke but is unavailable')
    if device.type=='cuda' and dtype=='bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA device does not support BF16')
    tolerance=3e-5 if dtype=='float32' else 5e-3
    torch.manual_seed(123)
    conf=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,
                     num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                     max_position_embeddings=128,attention_dropout=0.,tie_word_embeddings=False)
    model=Qwen2ForCausalLM(conf).to(device=device,dtype=torch_dtype).eval()
    model.config._attn_implementation='eager'
    pre=torch.tensor([[1,2,3,4]],device=device)
    def prefill():
        return model(input_ids=pre,use_cache=True,return_dict=True)
    def forward(tok,cache,pos,bias=None,*,native_default=False):
        mask=torch.zeros((1,1,1,cache.get_seq_length()+1),device=device,dtype=torch.float32)
        if bias is not None:
            assert len(bias)==mask.shape[-1]
            mask[0,0,0]=bias
        kwargs=dict(input_ids=torch.tensor([[tok]],device=device),past_key_values=cache,
                    position_ids=torch.tensor([[pos]],device=device),cache_position=torch.tensor([pos],device=device),
                    use_cache=True,return_dict=True)
        if not native_default:
            kwargs['attention_mask']=mask
        return model(**kwargs)
    with torch.no_grad():
        # Native default attention and an explicit all-zero 4D mask start from
        # separate prefills, so this also checks the public mask interface.
        native_cache=prefill().past_key_values
        zero_mask_cache=prefill().past_key_values
        native=forward(5,native_cache,4,native_default=True)
        zero_bias=torch.zeros(5,device=device,dtype=torch.float32)
        zero_mask=forward(5,zero_mask_cache,4,zero_bias)
        native_gap=float((native.logits-zero_mask.logits).abs().max())
        if native_gap>tolerance:
            raise AssertionError(f'native default and zero 4D mask differ: {native_gap}')

        # Patching the eager hook must preserve its output and call once per
        # decoder layer. A separate prefill keeps this comparison independent.
        probe_cache=prefill().past_key_values
        with AttentionLocalProbe() as probe:
            probe.latest_pair=(0,1)
            instrumented=forward(5,probe_cache,4,zero_bias)
            probe.latest_pair=None
            probe_count=len(probe.errors)
        instrument_gap=float((instrumented.logits-zero_mask.logits).abs().max())
        if instrument_gap>tolerance:
            raise AssertionError(f'local probe changed logits: {instrument_gap}')
        if probe_count!=conf.num_hidden_layers:
            raise AssertionError(f'expected {conf.num_hidden_layers} layer probes, got {probe_count}')

        def cache_before_query():
            out=prefill()
            for token,pos in ((5,4),(6,5)):
                bias=torch.zeros(out.past_key_values.get_seq_length()+1,device=device,dtype=torch.float32)
                out=forward(token,out.past_key_values,pos,bias)
            return out.past_key_values

        gaps={}
        for sign in (1,-1):
            logical_cache=cache_before_query()
            physical_cache=cache_before_query()
            logical_bias=torch.zeros(7,device=device,dtype=torch.float32)
            apply_pair_bias(logical_bias,4,0,sign)
            logical=forward(7,logical_cache,6,logical_bias)

            layers=getattr(physical_cache,'layers',None)
            if not layers:
                raise RuntimeError('DynamicCache API changed: missing .layers')
            dropped=5 if sign==1 else 4
            for layer in layers:
                if not hasattr(layer,'keys') or not hasattr(layer,'values'):
                    raise RuntimeError('DynamicCache changed: missing layer keys/values')
                if layer.keys.shape[-2]!=6:
                    raise AssertionError('unexpected physical cache sequence length')
                layer.keys=torch.cat((layer.keys[:,:,:dropped,:],layer.keys[:,:,dropped+1:,:]),dim=-2).contiguous()
                layer.values=torch.cat((layer.values[:,:,:dropped,:],layer.values[:,:,dropped+1:,:]),dim=-2).contiguous()
            physical_bias=torch.zeros(6,device=device,dtype=torch.float32)
            physical_bias[4]=math.log(2.)
            physical=forward(7,physical_cache,6,physical_bias)
            gaps[sign]=float((logical.logits-physical.logits).abs().max())
            if gaps[sign]>tolerance:
                raise AssertionError(f'logical mask does not match physical (2,0), sign={sign}: {gaps[sign]}')
    print('HF TINY INTEGRATION PASS: '
          f'device={device}, dtype={dtype}, tolerance={tolerance:g}, '
          f'native/zero-mask gap {native_gap:.3g}, probe gap {instrument_gap:.3g}, '
          f'{probe_count} layer probes, physical gaps +1={gaps[1]:.3g}/-1={gaps[-1]:.3g}')


def run_experiment(args):
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if transformers.__version__!='4.57.1':
        raise RuntimeError(f'Pinned transformers==4.57.1 required, found {transformers.__version__}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. This study requires the user GPU for the 7B checkpoint.')
    device=torch.device('cuda:0')
    prompts=load_prompt_list(Path(args.prompts))[:args.limit_prompts or None]
    if args.horizon>160 and not args.allow_long_horizon:
        raise ValueError('default protocol horizon capped at 160 for this experiment')
    if args.horizon<8:
        raise ValueError('need at least 8 output tokens to exercise temporal events')
    if args.paths_per_phase<1:
        raise ValueError('paths-per-phase must be positive')
    if args.limit_prompts<0:
        raise ValueError('limit-prompts must be nonnegative')
    out_path=Path(args.out_dir)
    out_path.mkdir(parents=True,exist_ok=False)
    model_path=args.model_path
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=False,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
                                               attn_implementation='eager',trust_remote_code=False,
                                               local_files_only=True).to(device).eval()
    if model.config.model_type!='qwen2':
        raise ValueError('The instrumented path is restricted to standard Qwen2 architecture')
    model.requires_grad_(False)
    model_config=model.config.to_dict()
    # HF configs often persist the local load path in this field.
    model_config.pop('_name_or_path',None)
    model_config.pop('name_or_path',None)
    expected_policies=['block','alternating','balanced_random']
    expected_phases=[1,-1]
    expected_replicates=list(range(args.paths_per_phase))
    manifest={
        'experiment':'L01_RealLLM_Temporal_Order_Pilot',
        'status':'RUNNING','model_type':model.config.model_type,
        'model_reference':public_model_reference(model_path),
        'model_params':sum(p.numel() for p in model.parameters()),
        'model_config':model_config,
        'model_declared_commit_hash':getattr(model.config,'_commit_hash',None),
        'source_sha256':{
            'prompts':hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
            'runner':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        'provenance':{
            'prompt_source_name':Path(args.prompts).name,
            'model_reference':public_model_reference(model_path),
            'model_declared_commit_hash':getattr(model.config,'_commit_hash',None),
            'weights_loaded_local_only':True,
        },
        'torch':torch.__version__,'transformers':transformers.__version__,
        'device_name':torch.cuda.get_device_name(),
        'horizon':args.horizon,'event_count':event_count(args.horizon),'random_seed':args.seed,
        'prompt_count':len(prompts),'paths_per_phase':args.paths_per_phase,'full_softmax':True,'temperature':1.,
        'expected_prompt_ids':[p['id'] for p in prompts],
        'expected_policy_ids':expected_policies,
        'expected_phases':expected_phases,
        'expected_replicate_ids':expected_replicates,
        'compression':'logical positive mass preserving (2,0) via 4d score mask',
        'physical_kv_memory_saving_measured':False,
        'sampling_law':'individually Q-sampled then compared to P on same history',
        'documented_precision':'BF16 weights/KV; float32 softmax/KL',
        'eos_handling':'fixed_horizon_continues_after_first_eos',
    }
    (out_path/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf8')
    rows=[]
    with AttentionLocalProbe() as probe, torch.inference_mode():
        for ip,prompt in enumerate(prompts):
            ids=encoded_prompt(tokenizer,prompt,device)
            n=ids.shape[1]
            if n+args.horizon>=getattr(model.config,'max_position_embeddings',32768):
                raise ValueError('context exceeds RoPE positional limit')
            # Pre-flight identity with a nontrivial custom 4D zero mask and two independent caches.
            identity=run_policy(model,tokenizer,ids,policy='identity',phase=1,horizon=8,
                                schedule_seed=stable_seed('schedule_v1',args.seed,prompt['id'],'identity'),
                                trajectory_seed=stable_seed('trajectory_v1',args.seed,prompt['id'],'identity'),
                                probe=probe)
            if identity['full_kl_q_to_p']>5e-5:
                raise AssertionError('nonzero full-cache identity KL under the 4D mask interface')
            for rep in range(args.paths_per_phase):
                # Use common inverse-CDF uniforms across both policies/phases.
                # Different repetitions also produce independent balanced-random schedules.
                schedule_seed=stable_seed('schedule_v1',args.seed,prompt['id'],rep)
                trajectory_seed=stable_seed('trajectory_v1',args.seed,prompt['id'],rep)
                for policy in ('block','alternating','balanced_random'):
                    for phase in (1,-1):
                        started=time.monotonic()
                        item=run_policy(model,tokenizer,ids,policy=policy,phase=phase,horizon=args.horizon,
                                        schedule_seed=schedule_seed,trajectory_seed=trajectory_seed,probe=probe)
                        item.update(prompt_id=prompt['id'],category=prompt['category'],prompt_len=int(n),
                                    replicate=rep,elapsed_seconds=time.monotonic()-started)
                        rows.append(item)
                        with (out_path/'rows.jsonl').open('a',encoding='utf8') as w:
                            w.write(json.dumps(item,ensure_ascii=False)+'\n')
                        print(json.dumps({'id':prompt['id'],'policy':policy,'phase':phase,'rep':rep,
                                          'KL':round(item['full_kl_q_to_p'],6),
                                          'local':round(item['local_attention_error_sq_mean'],6)},ensure_ascii=False),flush=True)
    expected_row_count=len(prompts)*len(expected_policies)*len(expected_phases)*len(expected_replicates)
    if len(rows)!=expected_row_count:
        raise AssertionError(f'completed {len(rows)} rows; expected {expected_row_count}')
    manifest['status']='COMPLETE';manifest['completed_rows']=len(rows)
    (out_path/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf8')
    return rows


def summarize(out_dir):
    import statistics
    path=Path(out_dir)
    manifest_path=path/'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('cannot summarize without manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('status')!='COMPLETE':
        raise ValueError('formal summary requires a COMPLETE manifest')
    expected_prompts=manifest.get('expected_prompt_ids')
    expected_policies=manifest.get('expected_policy_ids')
    expected_phases=manifest.get('expected_phases')
    expected_replicates=manifest.get('expected_replicate_ids')
    if not all(isinstance(x,list) and x for x in
               (expected_prompts,expected_policies,expected_phases,expected_replicates)):
        raise ValueError('manifest is missing a nonempty expected prompt/policy/phase/replicate grid')
    if len(set(expected_prompts))!=len(expected_prompts):
        raise ValueError('manifest expected prompt IDs are not unique')
    if set(expected_policies)!={'block','alternating','balanced_random'}:
        raise ValueError('manifest policy grid must contain block, alternating, balanced_random')
    if set(expected_phases)!={-1,1}:
        raise ValueError('manifest phase grid must contain exactly -1 and +1')
    if len(set(expected_replicates))!=len(expected_replicates):
        raise ValueError('manifest expected replicate IDs are not unique')
    if manifest.get('prompt_count')!=len(expected_prompts):
        raise ValueError('manifest prompt_count does not match expected prompt IDs')
    if manifest.get('paths_per_phase')!=len(expected_replicates):
        raise ValueError('manifest paths_per_phase does not match expected replicate IDs')
    horizon=manifest.get('horizon')
    if not isinstance(horizon,int) or horizon<1 or manifest.get('event_count')!=event_count(horizon):
        raise ValueError('manifest horizon/event_count is inconsistent')
    source_hashes=manifest.get('source_sha256')
    provenance=manifest.get('provenance')
    if not isinstance(source_hashes,dict) or not all(source_hashes.get(k) for k in ('prompts','runner')):
        raise ValueError('manifest is missing prompt/runner source hashes')
    if not isinstance(provenance,dict) or not provenance.get('model_reference'):
        raise ValueError('manifest is missing model provenance')

    raw=[json.loads(line) for line in (path/'rows.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    expected={(p,policy,phase,rep)
              for p in expected_prompts for policy in expected_policies
              for phase in expected_phases for rep in expected_replicates}
    observed=[]
    for row in raw:
        key=(row.get('prompt_id'),row.get('policy'),row.get('phase'),row.get('replicate'))
        observed.append(key)
        if key not in expected:
            raise ValueError(f'unexpected row outside manifest grid: {key}')
        if row.get('horizon')!=horizon:
            raise ValueError(f'row horizon does not match manifest for {key}')
        if row.get('events')!=manifest['event_count']:
            raise ValueError(f'row event count does not match manifest for {key}')
    from collections import Counter
    counts=Counter(observed)
    missing=[key for key in expected if counts[key]==0]
    duplicates=[key for key,count in counts.items() if count!=1]
    if missing or duplicates or len(raw)!=len(expected):
        raise ValueError(f'incomplete or duplicate manifest grid: missing={len(missing)}, duplicate={len(duplicates)}, '
                         f'rows={len(raw)}, expected={len(expected)}')
    if manifest.get('completed_rows')!=len(expected):
        raise ValueError('manifest completed_rows does not match the expected grid')

    groups={}
    for row in raw:
        key=(row['prompt_id'],row['category'],row['policy'])
        groups.setdefault(key,[]).append(row)
    collapsed={}
    for (p,cat,policy),rs in groups.items():
        reps={r['replicate'] for r in rs}
        if len(rs)!=2*len(reps) or any({x['phase'] for x in rs if x['replicate']==rep}!={-1,1} for rep in reps):
            raise ValueError('missing antiphase or replication / incomplete group')
        collapsed[(p,cat,policy)]={'kl':sum(x['full_kl_q_to_p'] for x in rs)/len(rs),
                                    'local':sum(x['local_attention_error_sq_mean'] for x in rs)/len(rs),
                                    'eos_count':sum(x['eos_count'] for x in rs),
                                    'replicates':len(reps)}
    comparisons=[]
    prompts=sorted({k[0] for k in collapsed})
    for p in prompts:
        entries={k[2]:v for k,v in collapsed.items() if k[0]==p}
        if set(entries)!={'block','alternating','balanced_random'}:
            raise ValueError('missing policy comparison')
        b,a,r=entries['block'],entries['alternating'],entries['balanced_random']
        comparisons.append({'prompt_id':p,
                            'category':next(k[1] for k in collapsed if k[0]==p),
                            'block_kl':b['kl'],'alt_kl':a['kl'],'random_kl':r['kl'],
                            'risk_ratio':b['kl']/max(a['kl'],1e-12),
                            'local_ratio':max(a['local'],b['local'])/max(1e-12,min(a['local'],b['local'])),
                            'block_local':b['local'],'alt_local':a['local'],'random_local':r['local'],
                            'eos_count':b['eos_count']+a['eos_count']+r['eos_count']})
    diffs=[z['block_kl']-z['alt_kl'] for z in comparisons]
    mean_diff=statistics.mean(diffs)
    ci=None
    if len(diffs)>=2:
        rng=random.Random(stable_seed('bootstrap_paired_prompt_v1','20261012'))
        bs=[statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs) for _ in range(3000)]
        bs.sort()
        ci=[bs[74],bs[2924]]
    summary={
      'n_prompts':len(comparisons),
      'n_positive_block_over_alt':sum(x['block_kl']>x['alt_kl'] for x in comparisons),
      'n_local_comparable_le1_25':sum(x['local_ratio']<=1.25 for x in comparisons),
      'n_joint_ratio_ge2_and_local_comparable':sum(x['risk_ratio']>=2 and x['local_ratio']<=1.25 for x in comparisons),
      'paired_mean_kl_difference_block_minus_alt':mean_diff,
      'paired_prompt_bootstrap95':ci,
      'policy_aggregate':{
        k:{'mean_kl':statistics.mean(x[f'{k}_kl'] for x in comparisons),
           'mean_local':statistics.mean(x[f'{k}_local'] for x in comparisons)}
        for k in ('block','alt','random')
      },
      'pairs':comparisons,
      'interpretation':'Nested sample paths and phases are averaged inside prompt. Bootstrap resamples prompts, not tokens. One model checkpoint is not multiple independent models. No performance/memory saving claim.'
    }
    (path/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='pairs'},indent=2))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-path',default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--prompts',default=str(Path(__file__).with_name('prompts.json')))
    ap.add_argument('--out-dir',default='runs/l01_llm_temporal_v0')
    ap.add_argument('--horizon',type=int,default=130)
    ap.add_argument('--seed',type=int,default=20261012)
    ap.add_argument('--limit-prompts',type=int,default=0)
    ap.add_argument('--paths-per-phase',type=int,default=2)
    ap.add_argument('--allow-long-horizon',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--hf-smoke-tiny',action='store_true')
    ap.add_argument('--hf-smoke-device',default='cpu')
    ap.add_argument('--hf-smoke-dtype',choices=('float32','bfloat16'),default='float32')
    ap.add_argument('--analyze-only',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        cpu_math_self_test();return
    if args.hf_smoke_tiny:
        hf_smoke_tiny(device=args.hf_smoke_device,dtype=args.hf_smoke_dtype);return
    if args.analyze_only:
        summarize(args.out_dir);return
    run_experiment(args)
    summarize(args.out_dir)

if __name__=='__main__':main()
