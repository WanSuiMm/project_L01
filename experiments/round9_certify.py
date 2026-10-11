#!/usr/bin/env python3
"""Round 9: reference-only, finite-history, path-local KV certificate.

Scientific status: rigorous real-arithmetic inequalities; float64 evaluation is
NOT an interval-arithmetic proof of a trained model. The per-history risk bound
uses only exact-model P reference trajectories and a given fixed summary, not
Q trajectories. Q is separately evaluated only for validation.

Use:
  python round9_certify.py --demo --T 4 --n 16 --m 15
  python round9_certify.py --checkpoint PATH/checkpoint.pt --project-root /path/to/project_L01 --T 3 --n 16 --m 15
"""
import argparse, itertools, sys, json, math, os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Deterministic CPU computations. Avoid TF32 and kernel-level stochasticity.
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
DTYPE = torch.float64


class DecoderBlock(nn.Module):
    def __init__(self, d, ff=2):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d); self.k_proj=nn.Linear(d,d); self.v_proj=nn.Linear(d,d)
        self.out_proj=nn.Linear(d,d)
        self.norm_ff=nn.LayerNorm(d); self.ff_in=nn.Linear(d, ff*d); self.ff_out=nn.Linear(ff*d,d)


class TinyCausalDecoder(nn.Module):
    def __init__(self,vocab_size=3,d_model=16,n_layers=2,max_context=80,ff_multiplier=2):
        super().__init__()
        self.vocab_size=vocab_size;self.d_model=d_model;self.n_layers=n_layers;self.max_context=max_context
        self.ff_multiplier=ff_multiplier
        self.token_embedding=nn.Embedding(vocab_size,d_model)
        self.position_embedding=nn.Embedding(max_context,d_model)
        self.layers=nn.ModuleList([DecoderBlock(d_model,ff_multiplier) for _ in range(n_layers)])
        self.final_norm=nn.LayerNorm(d_model);self.lm_head=nn.Linear(d_model,vocab_size,bias=False)


def copy_model_from_checkpoint(checkpoint,project_root):
    # Import original source only for schema confirmation. Use the equivalent class above
    # to run in float64 with explicit attention and easily readable traces.
    root=Path(project_root)
    if not (root/'experiments'/'round6_transformer.py').exists():
        raise FileNotFoundError('project-root must contain experiments/round6_transformer.py')
    sys.path.insert(0,str(root/'experiments'))
    from round6_transformer import TinyCausalDecoder as Original
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    baseline=Original(**payload['model'])
    baseline.load_state_dict(payload['state_dict'], strict=True)
    config=payload['model']
    model=TinyCausalDecoder(**config).double()
    model.load_state_dict(baseline.state_dict(),strict=True)
    model.eval()
    return model,payload


def demo_model():
    torch.manual_seed(8107)
    model=TinyCausalDecoder().double().eval()
    with torch.no_grad():
        # Nonzero two-layer Transformer with small attention/MLP updates.
        # Not a trained checkpoint; strictly a verifier smoke fixture.
        for layer in model.layers:
            for module in (layer.q_proj,layer.k_proj,layer.v_proj,layer.out_proj,
                           layer.ff_in,layer.ff_out):
                module.weight.mul_(0.16)
                if module.bias is not None: module.bias.mul_(0.16)
        model.position_embedding.weight.mul_(0.08)
        model.lm_head.weight.mul_(0.36)
    return model


def norm(v): return float(torch.linalg.vector_norm(v).item())


def opnorm_linear(m): return float(torch.linalg.matrix_norm(m.weight,ord=2).item())


def layernorm_chord_lip(ln, ref, radius):
    """Rigorous real-arithmetic gain on the radius-ball around ref."""
    d=ref.numel()
    c=ref-ref.mean()
    std=norm(c)/math.sqrt(d)
    lower=max(0.,std-radius/math.sqrt(d))
    return float(ln.weight.detach().abs().max())/math.sqrt(lower*lower+ln.eps)


def layernorm_global_range(ln):
    d=ln.weight.numel()
    return math.sqrt(d)*float(ln.weight.detach().abs().max())+norm(ln.bias.detach())


def op_constants(model):
    out=[]
    for l in model.layers:
        out.append({z:opnorm_linear(getattr(l,z)) for z in ['q_proj','k_proj','v_proj','out_proj','ff_in','ff_out']})
    return out


def clone_cache(caches):
    return [(x[0].clone(),x[1].clone(),x[2].clone()) for x in caches]


@torch.no_grad()
def prefill(model,prefix):
    """Dense-prefill: return exactly the keys/values cached by the original architecture."""
    prefix=torch.as_tensor(prefix,dtype=torch.long)
    positions=torch.arange(len(prefix),dtype=torch.long)
    hidden=model.token_embedding(prefix)+model.position_embedding(positions)
    caches=[]
    for l in model.layers:
        normalized=l.norm_attn(hidden)
        q=l.q_proj(normalized); k=l.k_proj(normalized);v=l.v_proj(normalized)
        scores=q@k.T / math.sqrt(model.d_model)
        idx=torch.arange(len(prefix))
        scores[idx[:,None]<idx[None,:]]=-torch.inf
        p=torch.softmax(scores,dim=-1)
        a=p@v
        y=hidden+l.out_proj(a)
        hidden=y+l.ff_out(F.gelu(l.ff_in(l.norm_ff(y))))
        caches.append((k.clone(),v.clone(),torch.ones(len(prefix),dtype=DTYPE)))
    return caches


def merge_nearest_pair(full, n, m):
    if m!=n-1: raise ValueError('Current positive-cluster demo uses exactly m=n-1.')
    out=[]; selected=[]
    for k,v,w in full:
        x=torch.cat((k,v),dim=-1)
        d=((x[:,None,:]-x[None,:,:])**2).sum(-1)
        d.fill_diagonal_(float('inf'))
        i,j=np.unravel_index(int(d.argmin().item()),(n,n))
        rest=[ii for ii in range(n) if ii not in (i,j)]
        # Pair average as new synthetic KV record, weight 2; all others exact.
        kk=torch.cat((k[rest],.5*(k[i:i+1]+k[j:j+1])),dim=0)
        vv=torch.cat((v[rest],.5*(v[i:i+1]+v[j:j+1])),dim=0)
        ww=torch.cat((torch.ones(len(rest),dtype=DTYPE),torch.tensor([2.],dtype=DTYPE)))
        out.append((kk,vv,ww));selected.append([int(i),int(j)])
    return out,selected


def merge_agglomerative(full,n,m):
    """Positive cardinality-weighted mean representatives; prefix-only choice."""
    if not 1 <= m < n:raise ValueError('agglomerative requires 1<=m<n')
    result=[]; metadata=[]
    for keys,values,_ in full:
        groups=[[i] for i in range(n)]
        while len(groups)>m:
            best=None;pair=None
            means=[torch.cat((keys[ids].mean(0),values[ids].mean(0))) for ids in groups]
            for i in range(len(groups)):
                for j in range(i+1,len(groups)):
                    d=float(torch.sum((means[i]-means[j])**2))
                    if best is None or d<best:best=d;pair=(i,j)
            i,j=pair
            groups[i]=groups[i]+groups[j];del groups[j]
        kk=torch.stack([keys[ids].mean(0) for ids in groups]);vv=torch.stack([values[ids].mean(0) for ids in groups])
        ww=torch.tensor([len(ids) for ids in groups],dtype=DTYPE)
        result.append((kk,vv,ww));metadata.append(groups)
    return result,metadata


def merge_greedy_pairs(full,n,m):
    """Deterministic positive mass-two representative for each closest pair."""
    if n%2 or m!=n//2:raise ValueError('greedy_pairs requires n even, m=n/2')
    results=[];metadata=[]
    for k,v,w in full:
        x=torch.cat((k,v),dim=-1)
        remain=list(range(n));parts=[]
        while remain:
            best=None;bi=bj=None
            for ia,i in enumerate(remain):
                for j in remain[ia+1:]:
                    val=float(torch.sum((x[i]-x[j])**2))
                    if best is None or val<best:best=val;bi=i;bj=j
            parts.append((bi,bj));remain.remove(bi);remain.remove(bj)
        kk=torch.stack([.5*(k[i]+k[j]) for i,j in parts]); vv=torch.stack([.5*(v[i]+v[j]) for i,j in parts]); ww=torch.full((m,),2.,dtype=DTYPE)
        results.append((kk,vv,ww));metadata.append(parts)
    return results,metadata


def weighted_subset(full,n,m,seed=101):
    if m<1 or m>=n:raise ValueError('m<n required')
    rng=np.random.default_rng(seed)
    out=[]
    for k,v,w in full:
        ids=sorted(rng.choice(n,size=m,replace=False).tolist())
        out.append((k[ids],v[ids],torch.full((m,),n/m,dtype=DTYPE)))
    return out,[]


@torch.no_grad()
def step_trace(model,cache,token,position):
    x=model.token_embedding(torch.tensor(token))+model.position_embedding(torch.tensor(position))
    new=[]; traces=[]
    for l,(pk,pv,pw) in zip(model.layers,cache):
        pre=x.clone(); normed=l.norm_attn(x)
        q=l.q_proj(normed);k=l.k_proj(normed);v=l.v_proj(normed)
        kk=torch.cat((pk,k[None]),0);vv=torch.cat((pv,v[None]),0); ww=torch.cat((pw,torch.ones(1,dtype=DTYPE)),0)
        score=(kk@q)/math.sqrt(model.d_model)
        attn=torch.softmax(score+torch.log(ww),dim=0)
        a=(attn[:,None]*vv).sum(0)
        y=x+l.out_proj(a)
        x=y+l.ff_out(F.gelu(l.ff_in(l.norm_ff(y))))
        traces.append(dict(pre=pre,mid=y,post=x,q=q,k=k,v=v,score=score,attn=attn,
                           a=a,keys=kk,values=vv,weights=ww))
        new.append((kk,vv,ww))
    logits=model.lm_head(model.final_norm(x))
    return logits,new,traces


@torch.no_grad()
def reference_trajectory(model,full_cache,history,cont_id):
    cache=clone_cache(full_cache)
    traces=[];logits=[]
    for j, token in enumerate([cont_id,*history]):
        z,cache,record=step_trace(model,cache,token,len(full_cache[0][0])+j)
        traces.append(record);logits.append(z)
    return traces,logits



def ref_attention_on_summary(model,l,s,records,summary_cache):
    trace=records[s][l]; kprefix,vprefix,wprefix=summary_cache[l]
    k_suffix=torch.stack([records[j][l]['k'] for j in range(s+1)])
    v_suffix=torch.stack([records[j][l]['v'] for j in range(s+1)])
    k_all=torch.cat((kprefix,k_suffix),dim=0)
    v_all=torch.cat((vprefix,v_suffix),dim=0)
    w_all=torch.cat((wprefix,torch.ones(s+1,dtype=DTYPE)),0)
    q=trace['q']/model.d_model**.25
    k_scaled=k_all/model.d_model**.25
    scores=k_scaled@q + w_all.log()
    p=torch.softmax(scores,dim=0)
    A=(p[:,None]*v_all).sum(0)
    return k_scaled,v_all,w_all,q,p,A


def softmax_perturb_bound(k_scaled,v_all,q_ref,p_ref,A_ref,kappa,nu,zeta,global_value_bound):
    """Certified endpoint interval bound; common score shift removed via key centering.
    Bounds |delta(q)|<=zeta, |delta(k_i)|<=kappa_i, |delta(v_i)|<=nu_i.
    For any admissible perturbations, returns B s.t. ||A'-A|| <= B.
    """
    center=(p_ref[:,None]*k_scaled).sum(0)
    dkey=torch.linalg.vector_norm(k_scaled-center[None],dim=-1)
    U=(zeta*dkey+(norm(q_ref)+zeta)*kappa).clamp_min(0)
    vmax=float(U.max())
    maxv=float(torch.linalg.vector_norm(v_all,dim=-1).max())
    if vmax>60:
        # Both old/new attention outputs remain in value convex hull enlarged
        # by maximum suffix value displacement; avoid numerical exponential.
        return global_value_bound+maxv,{'score_error_max':vmax,'method':'bounded_value_fallback'}
    Eminus=float((p_ref*torch.exp(-U)).sum())
    Eplus=float((p_ref*torch.exp(U)).sum())
    # Bounds on all possible perturbed softmax probabilities.
    lower=p_ref*torch.exp(-U)/Eplus
    upper=p_ref*torch.exp(U)/Eminus
    range_diff=torch.maximum(p_ref-lower,upper-p_ref).clamp_min(0)
    dispersion=torch.linalg.vector_norm(v_all-A_ref[None],dim=-1)
    term_value=float(torch.sum(upper*nu))
    term_probs=float(torch.sum(range_diff*dispersion))
    # Independent global convex-hull fallback is also rigorous.
    value_cap=global_value_bound+maxv
    bound=min(term_value+term_probs,value_cap)
    return bound,{'score_error_max':vmax,'value_term':term_value,'probability_term':term_probs,
                  'method':'local_probability_intervals'}


def bound_all_steps(model,full_cache,summary_cache,records,ops):
    L=len(model.layers);t_count=len(records); n=len(full_cache[0][0]);d=model.d_model
    # E[l][s] bounds hidden discrepancy after layer l at suffix position s.
    E=[[0.]*t_count for _ in range(L+1)]
    forcing=[[0.]*t_count for _ in range(L)]
    soft_perturb=[[0.]*t_count for _ in range(L)]
    tube_stats=[]
    for l,layer in enumerate(model.layers):
        c=ops[l]
        for t in range(t_count):
            ref=records[t][l]
            ein=E[l][t]
            centered_ref=norm(ref['pre']-ref['pre'].mean())
            tube_stats.append(('attn',l,t,ein/max(centered_ref,1e-30)))
            zeta=(c['q_proj']/d**.25)*layernorm_chord_lip(layer.norm_attn,ref['pre'],ein)*ein
            kscaled,vals,weights,q,p,A=ref_attention_on_summary(model,l,t,records,summary_cache)
            original=ref['a']; forc=norm(A-original)
            npre=len(summary_cache[l][0]);nall=len(vals)
            kappa=torch.zeros(nall,dtype=DTYPE)
            nu=torch.zeros(nall,dtype=DTYPE)
            for s in range(t+1):
                x=records[s][l]['pre'];prev=E[l][s]
                ln_gain=layernorm_chord_lip(layer.norm_attn,x,prev)
                kappa[npre+s]=c['k_proj']/d**.25 *ln_gain*prev
                nu[npre+s]=c['v_proj']*ln_gain*prev
            global_vbound=c['v_proj']*layernorm_global_range(layer.norm_attn)+norm(layer.v_proj.bias)
            pert,diag=softmax_perturb_bound(kscaled,vals,q,p,A,kappa,nu,zeta,global_vbound)
            ba=min(forc+pert,2*global_vbound)
            dy=ein+c['out_proj']*ba
            # Exact architecture: y = x + out_proj(a); h = y + FF(y).
            centered_ff=norm(ref['mid']-ref['mid'].mean())
            tube_stats.append(('ff',l,t,dy/max(centered_ff,1e-30)))
            lnff=layernorm_chord_lip(layer.norm_ff,ref['mid'],dy)
            ff_gain=c['ff_out']*1.13*c['ff_in']*lnff
            ff_input_bound=c['ff_in']*layernorm_global_range(layer.norm_ff)+norm(layer.ff_in.bias)
            ff_diff_cap=2*c['ff_out']*ff_input_bound
            eout=dy+min(ff_gain*dy,ff_diff_cap)
            E[l+1][t]=eout
            forcing[l][t]=forc;soft_perturb[l][t]=pert
    final_ref=records[-1][-1]['post']
    r=E[-1][-1]
    tube_stats.append(('final',L,t_count-1,r/max(norm(final_ref-final_ref.mean()),1e-30)))
    gain=layernorm_chord_lip(model.final_norm,final_ref,r)
    W=model.lm_head.weight
    # Row-contrast norm exactly respects softmax gauge invariance.
    contrasts=W-W.mean(0,keepdim=True)
    op_contrast=float(torch.linalg.matrix_norm(contrasts,ord=2))
    normed_delta=min(gain*r,2*layernorm_global_range(model.final_norm))
    op_readout=op_contrast*gain
    Rlog=op_contrast*normed_delta
    max_pair=max(norm(W[i]-W[j]) for i in range(W.shape[0]) for j in range(i))
    osc=max_pair*normed_delta
    # KL <= osc^2/8, from Bregman log-partition Hessian Var<=osc^2/4.
    one_step=osc**2/8
    # KL <= ||centered Delta logits||^2/4 also strict.
    one_step=min(one_step,Rlog**2/4)
    # Improved Fisher-weighted bound at exact reference distribution.
    # Need original reference logits computed by caller; provided separately.
    return {
        'E_end':E[-1][-1], 'E_all':E, 'forcing':forcing, 'soft_perturb':soft_perturb,
        'LN_final_gain':gain, 'LN_final_delta_bound':normed_delta, 'output_osc_bound':osc,
        'max_tube_fraction':max(x[3] for x in tube_stats),
        'tube_intersects_zero_variance':sum(x[3]>=1 for x in tube_stats), 'local_bound_pre_fisher':one_step,
    }


def certified_conditional_bound(model,model_output,entry):
    """Fisher sharpening uses exact P conditional probabilities but no Q."""
    p=torch.softmax(model_output,dim=0)
    W=model.lm_head.weight
    centred=W-(p[:,None]*W).sum(0)[None]
    gram=centred.T@(p[:,None]*centred)
    eigmax=max(0.,float(torch.linalg.eigvalsh(gram).max()))
    R=entry['E_end'];LN=entry['LN_final_gain'];norm_delta=entry['LN_final_delta_bound']
    osc=entry['output_osc_bound']
    if osc>=700: fisher=float('inf')
    else: fisher=.5*math.exp(osc)*eigmax*norm_delta**2
    # A model-dependent universal cap given known exact-reference P.
    universal=float(-torch.log(p.min()))
    both_bound=min(entry['local_bound_pre_fisher'],fisher)
    qp_bound=min(both_bound,universal)  # universal cap is Q||P ONLY, not P||Q
    return dict(bound=float(qp_bound),bound_PQ=float(both_bound),fisher=float(fisher),reference_cap=universal,
                simple=entry['local_bound_pre_fisher'],p=p.tolist())


@torch.no_grad()
def eval_exact_pair(model,full_cache,summary_cache,history,cont_id):
    fp=clone_cache(full_cache); fq=clone_cache(summary_cache)
    for t,token in enumerate([cont_id,*history]):
        zp,fp,_=step_trace(model,fp,token,len(full_cache[0][0])+t)
        zq,fq,_=step_trace(model,fq,token,len(full_cache[0][0])+t)
    return torch.softmax(zp,dim=0),torch.softmax(zq,dim=0)


def run_cert(model,prefix,cont_id,T,m,summary_kind,include_exact_q=True,verbose=False):
    n=len(prefix);assert m<n
    if n+T>model.max_context:raise ValueError('Model max context too short')
    with torch.no_grad():
        full_cache=prefill(model,prefix)
        if summary_kind=='closest_pair':summary,pairs=merge_nearest_pair(full_cache,n,m)
        elif summary_kind=='greedy_pairs':summary,pairs=merge_greedy_pairs(full_cache,n,m)
        elif summary_kind=='agglomerative':summary,pairs=merge_agglomerative(full_cache,n,m)
        else: summary,pairs=weighted_subset(full_cache,n,m)
        ops=op_constants(model)
        tree={}
        for length in range(T):
            for history in itertools.product(range(model.vocab_size),repeat=length):
                records,logits=reference_trajectory(model,full_cache,history,cont_id)
                entry=bound_all_steps(model,full_cache,summary,records,ops)
                entry.update(certified_conditional_bound(model,logits[-1],entry))
                if include_exact_q:
                    p,q=eval_exact_pair(model,full_cache,summary,history,cont_id)
                    kl_qp=float((q*(torch.log(q)-torch.log(p))).sum().clamp_min(0))
                    kl_pq=float((p*(torch.log(p)-torch.log(q))).sum().clamp_min(0))
                    entry['exact_KL_QP']=kl_qp;entry['exact_KL_PQ']=kl_pq
                    entry['q']=q.tolist()
                    # loose floating-point handling; treat failing invariants as errors.
                    if kl_qp>entry['bound']+1e-10 or kl_pq>entry['bound_PQ']+1e-10:
                        raise AssertionError(f'Invalid local certificate at {history}: {kl_qp}, {kl_pq} > {entry["bound"]}')
                tree[history]=entry
        # Compute reference and compressed complete law + finite-branch pathwise sums.
        exact_qp=exact_pq=reverse_upper=0.
        E_p_risk=0.
        leaves=[]
        for leaf in itertools.product(range(model.vocab_size),repeat=T):
            ppath=qpath=1.
            risk=0.
            risk_PQ=0.
            for t in range(T):
                h=leaf[:t];a=leaf[t];data=tree[h]
                ppath*=data['p'][a]
                if include_exact_q:qpath*=data['q'][a]
                risk+=data['bound']
                risk_PQ+=data['bound_PQ']
            leaves.append((leaf,ppath,qpath,risk,risk_PQ))
        assert abs(sum(x[1] for x in leaves)-1)<1e-7
        if include_exact_q:assert abs(sum(x[2] for x in leaves)-1)<1e-7
        # D(P||Q) <= E_P R, and D(Q||P) <= E_Q R.
        E_p_risk=sum(p*rP for _,p,_,r,rP in leaves)
        maxpath=max(z[3] for z in leaves)
        if include_exact_q:
            E_q_risk=sum(q*r for _,p,q,r,_ in leaves)
            # Use KL chain rule with nonnegative conditional terms to avoid
            # cancellation in nearly identical full-sequence probabilities.
            for leaf,p,q,_,_ in leaves:
                q_path_kl=sum(tree[leaf[:t]]['exact_KL_QP'] for t in range(T))
                p_path_kl=sum(tree[leaf[:t]]['exact_KL_PQ'] for t in range(T))
                exact_qp+=q*q_path_kl
                exact_pq+=p*p_path_kl
        else: E_q_risk=None
        # Relative entropy self-bound for unknown Q:
        # (lambda-1) D(Q||P) <= log E_P exp(lambda R)
        lambdas=[1.001,1.01,1.05,1.1,1.2,1.4,1.6,2,3,4,6,8,12,20,35,60,100,200,500,1000]
        pv=np.array([x[1] for x in leaves]);rr=np.array([x[3] for x in leaves]);
        def logsumexp_np(v):
            z=np.max(v);return z+np.log(np.sum(np.exp(v-z)))
        dv=[]
        for lam in lambdas:
            mom=logsumexp_np(np.log(pv)+lam*rr)
            dv.append((float(mom/(lam-1)),lam))
        dv_best=min(dv)
        best=min(maxpath,dv_best[0])
        # Additional coarse full-support KL bound computed only from reference P.
        baseline_cap=sum(max(tree[h]['reference_cap'] for h in tree if len(h)==t) for t in range(T))
        summary_stats={
            'status':'FLOAT64_REFERENCE_LOCAL_BOUNDS_NOT_INTERVAL_CERTIFIED',
            'model_kind':getattr(model,'_source_kind','unlabeled'),
            'n':n,'m':m,'T':T,'vocab':model.vocab_size,'histories':len(tree),'full_sequence_leaves':len(leaves),
            'summary_kind':summary_kind,'pair_merges':pairs,
            'cert_QP_upper':best,'cert_PQ_upper':E_p_risk,
            'certificate_raw_max_path':maxpath,'certificate_DV_bound':dv_best[0],
            'certificate_DV_lambda':dv_best[1],
            'reference_universal_cap':baseline_cap,
            'nonvacuous_vs_reference_cap':best<baseline_cap,
            'largest_conditional_local_bound':max(x['bound'] for x in tree.values()),
            'largest_hidden_radius':max(max(max(row) for row in v['E_all']) for v in tree.values()),
            'largest_tube_fraction':max(v['max_tube_fraction'] for v in tree.values()),
            'tube_zero_variance_intersections':sum(v['tube_intersects_zero_variance'] for v in tree.values()),
            'largest_reference_only_forcing':max(max(max(r) for r in x['forcing']) for x in tree.values()),
            'exact_KL_QP':exact_qp if include_exact_q else None,
            'exact_KL_PQ':exact_pq if include_exact_q else None,
            'exact_Q_weighted_pathwise_envelope':E_q_risk if include_exact_q else None,
            'ratio_QP_upper_over_actual':best/max(exact_qp,1e-30) if include_exact_q else None,
        }
        return summary_stats,tree


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--demo',action='store_true')
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--fixture-checkpoint',type=Path,help='standalone CPU toy fixture only, not the user checkpoint')
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    ap.add_argument('--n',type=int,default=16)
    ap.add_argument('--m',type=int,default=15)
    ap.add_argument('--T',type=int,default=4)
    ap.add_argument('--summary',choices=['closest_pair','greedy_pairs','agglomerative','weighted_subset'],default='closest_pair')
    ap.add_argument('--seed',type=int,default=20261016)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if sum(bool(x) for x in (args.demo,args.checkpoint,args.fixture_checkpoint))>1:ap.error('choose only one model source')
    if args.T>7:ap.error('Exact full-softmax tree is exponential; cap T<=7')
    if args.demo or (args.checkpoint is None and args.fixture_checkpoint is None):
        model=demo_model();model._source_kind='untrained_deterministic_demo'
        prefix=np.random.default_rng(args.seed).integers(0,2,size=args.n).tolist()
        cont_id=2
    elif args.fixture_checkpoint is not None:
        payload=torch.load(args.fixture_checkpoint,map_location='cpu',weights_only=True)
        model=TinyCausalDecoder(**payload['model']).double().eval()
        model.load_state_dict(payload['state_dict'],strict=True)
        model._source_kind='new_cpu_trained_toy_fixture_not_original'
        prefix=np.random.default_rng(args.seed).integers(0,2,size=args.n).tolist()
        cont_id=2
    else:
        model,payload=copy_model_from_checkpoint(args.checkpoint,args.project_root)
        model._source_kind='user_existing_round6_checkpoint_float64_replay'
        # Use exactly the frozen Round 6 histogram episode sampler: prefix of 0/1 tokens.
        rng=np.random.default_rng(args.seed)
        theta=float(rng.beta(2,2))
        prefix=rng.binomial(1,theta,size=args.n).tolist()
        cont_id=int(payload['task']['cont_id'])
        if model.n_layers!=2 or model.vocab_size!=3:
            raise ValueError('This focused audit expects the two-layer, three-token Round 6 histogram checkpoint')
    if args.summary=='closest_pair' and args.m!=args.n-1:
        ap.error('closest_pair requires --m == --n - 1')
    if args.summary=='greedy_pairs' and args.m*2!=args.n:
        ap.error('greedy_pairs requires --m == --n//2')
    result,tree=run_cert(model,prefix,cont_id,args.T,args.m,args.summary)
    result['prefix']=prefix
    print(json.dumps(result,indent=2,allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
