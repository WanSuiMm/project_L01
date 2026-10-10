#!/usr/bin/env python3
"""Deterministic numerical checks for the Round 7 proof draft.

No model training; no empirical claim about trained Transformers; no test can
replace a proof. GS-Thin's sub-Gaussian guarantee is an imported theorem and is
not established by this script. Dependencies: numpy, scipy.
"""
from __future__ import annotations
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261010)
SIGMA_PRIME_1 = float(expit(1.0) * (1.0 - expit(1.0)))
TOL = 2e-10


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def bern_kl(a, b):
    """KL(Ber(sigmoid(a)) || Ber(sigmoid(b))), in float64."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ans = expit(a) * (a-b) - np.logaddexp(0.0, a) + np.logaddexp(0.0, b)
    check(bool(np.all(ans >= -1e-12)), "materially negative Bernoulli KL")
    return np.maximum(ans, 0.0)


def h_s(n: int, horizon: int) -> tuple[float, float]:
    f = n / (n + np.arange(1, horizon+1, dtype=float))
    return n*horizon/(n+horizon), float(f @ f)


def histogram_law_kl(n: int, mu: float, muhat: float, horizon: int) -> tuple[float, float]:
    """Exact DP for a legal binary, one-layer, uniform-softmax decoder.

Common start token is +1. Future values are actual generated +/-1 tokens,
not artificially zero. DP state is the number of +1 generated tokens so far.
Returns KL(Q_R||P), KL(P||Q_R) for one fixed prefix summary.
"""
    p_state = np.array([1.0]); q_state = np.array([1.0])
    forward = reverse = 0.0
    for t in range(1, horizon+1):
        count_sum = 1.0 + 2*np.arange(t) - (t-1)
        a = (n*mu + count_sum)/(n+t)
        b = (n*muhat + count_sum)/(n+t)
        p = expit(a); q = expit(b)
        forward += float(q_state @ bern_kl(b, a))
        reverse += float(p_state @ bern_kl(a, b))
        pp = np.zeros(t+1); qq = np.zeros(t+1)
        pp[:-1] += p_state*(1-p); pp[1:] += p_state*p
        qq[:-1] += q_state*(1-q); qq[1:] += q_state*q
        p_state, q_state = pp, qq
    check(abs(p_state.sum()-1) < TOL and abs(q_state.sum()-1) < TOL, "DP normalization")
    return forward, reverse


def analytic_checks() -> dict:
    pairs = RNG.uniform(-1, 1, (600, 2))
    kl = bern_kl(pairs[:, 0], pairs[:, 1])
    d2 = np.square(pairs[:, 0]-pairs[:, 1])
    check(bool(np.all(kl + TOL >= SIGMA_PRIME_1*d2/2)), "scalar KL lower")
    check(bool(np.all(kl <= d2/8 + TOL)), "scalar KL upper")
    lower_rows, uniform_rows = [], []
    all_integer_choices = 0
    worst_lower_ratio = math.inf
    for n in [8, 16, 32, 64, 128]:
        for m in [2**j for j in range(int(math.log2(n)))]:
            plus = n//(2*m); minus = n-plus; mu = 1/m-1
            all_means = []
            for s in range(1, m+1):
                for k in range(max(0, s-minus), min(s, plus)+1):
                    muh = 2*k/s-1
                    check(abs(muh-mu)+TOL >= 1/m, "thinning grid lower for s<=m")
                    all_integer_choices += 1; all_means.append(muh)
            for horizon in [1, 4, 16, 64]:
                h, ss = h_s(n, horizon)
                check(h/3 - TOL <= ss <= h+TOL, "Hardy forcing finite sum")
                # All possible m-record counts, including the best possible choice.
                risks = []
                for k in range(max(0, m-minus), min(m, plus)+1):
                    muh = 2*k/m-1
                    fw, rv = histogram_law_kl(n, mu, muh, horizon)
                    lo = SIGMA_PRIME_1*ss*(muh-mu)**2/2
                    hi = ss*(muh-mu)**2/8
                    check(lo-TOL <= fw <= hi+TOL, "minimax forward pointwise bracket")
                    check(lo-TOL <= rv <= hi+TOL, "minimax reverse pointwise bracket")
                    risks.append((fw, rv))
                bestfw = min(z[0] for z in risks); bestrv = min(z[1] for z in risks)
                bound = SIGMA_PRIME_1*h/(6*m*m)
                check(bestfw+TOL >= bound and bestrv+TOL >= bound, "minimax h/m^2 lower")
                worst_lower_ratio = min(worst_lower_ratio, bestfw/(h/m**2), bestrv/(h/m**2))
                lower_rows.append(dict(n=n,m=m,T=horizon,h=h,S=ss,
                    best_equal_weight_forward_kl=bestfw,best_equal_weight_reverse_kl=bestrv,
                    proved_lower=bound,feasible_m_record_counts=len(risks)))
                # Separate balanced-prefix uniform-sampling comparison.
                ks = np.arange(max(0,m-n//2), min(m,n//2)+1)
                prob = hypergeom.pmf(ks, n, n//2, m)
                check(abs(prob.sum()-1) < TOL, "hypergeometric probabilities")
                vf = (n-m)/(m*(n-1))
                qmean = 2*ks/m-1
                check(abs(float(prob @ qmean**2)-vf) < TOL, "finite population variance")
                expected_fw = expected_rv = 0.0
                for k, pr in zip(ks, prob):
                    fw, rv = histogram_law_kl(n, 0.0, 2*k/m-1, horizon)
                    expected_fw += pr*fw; expected_rv += pr*rv
                lo, hi = SIGMA_PRIME_1*ss*vf/2, ss*vf/8
                check(lo-TOL <= expected_fw <= hi+TOL, "uniform forward bracket")
                check(lo-TOL <= expected_rv <= hi+TOL, "uniform reverse bracket")
                uniform_rows.append(dict(n=n,m=m,T=horizon,h=h,S=ss,
                    variance=vf,mean_forward_kl=expected_fw,mean_reverse_kl=expected_rv,
                    lower=lo,upper=hi))
    for name, rows in [('equal_weight_lower.csv',lower_rows),('uniform_sampling_risk.csv',uniform_rows)]:
        with (ROOT/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(scalar_kl_checks=len(pairs), integer_summary_choices=all_integer_choices,
        minimax_grid_cases=len(lower_rows), uniform_sampling_grid_cases=len(uniform_rows),
        min_best_risk_over_h_div_m_squared=worst_lower_ratio,
        theoretical_min_coefficient=SIGMA_PRIME_1/6)


def tree_transfer_checks() -> dict:
    """Exhaustive checks of the abstract energy/entropy transfer lemma.

A random seed is a finite Rademacher vector; each node's logit perturbation
is linear in that seed. This checks the probabilistic lemma, not KV compression.
"""
    cases = 0; forward_cases = 0; worst_mgf_ratio = 0.0
    max_chain_gap = 0.0
    for horizon in [3,4,5,6]:
        for dim in [2,3,4]:
            for scale in [.08,.4,1.2]:
                node_n = 2**horizon-1; leaf_n = 2**horizon
                ref = RNG.normal(0,.5,node_n)
                coeff = RNG.normal(size=(node_n,dim))*scale/math.sqrt(dim*horizon)
                leaves = np.array(list(itertools.product([0,1],repeat=horizon)))
                nodes = np.zeros((leaf_n,horizon),dtype=int)
                for i,y in enumerate(leaves):
                    node=0
                    for t,b in enumerate(y):
                        nodes[i,t]=node;node=2*node+1+int(b)
                trace = np.square(coeff[nodes]).sum(axis=(1,2))/8
                V = float(trace.max());lam=1/(4*V)
                seeds=np.array(list(itertools.product([-1.,1.],repeat=dim)))
                base = ref[nodes]
                logp = (leaves*base-np.logaddexp(0,base)).sum(axis=1)
                p = np.exp(logp);check(abs(p.sum()-1)<TOL,'reference tree normalization')
                dplus=[];dminus=[];uall=[]
                for seed in seeds:
                    delta=coeff@seed; alt=base+delta[nodes]
                    logq=(leaves*alt-np.logaddexp(0,alt)).sum(axis=1);q=np.exp(logq)
                    check(abs(q.sum()-1)<TOL,'seed tree normalization')
                    fw=float(q@(logq-logp));rv=float(p@(logp-logq))
                    fpath=bern_kl(alt,base).sum(axis=1);rpath=bern_kl(base,alt).sum(axis=1)
                    max_chain_gap=max(max_chain_gap,abs(fw-q@fpath),abs(rv-p@rpath))
                    U=np.square(delta[nodes]).sum(axis=1)/8
                    check(bool(np.all(fpath<=U+TOL) and np.all(rpath<=U+TOL)),'energy envelope')
                    check((lam-1)*fw <= logsumexp(logp+lam*U)+TOL,'entropy absorption algebra')
                    dplus.append(fw);dminus.append(rv);uall.append(U)
                dplus=np.array(dplus);dminus=np.array(dminus);uall=np.array(uall)
                mgf=np.exp(uall/(4*V)).mean(axis=0)
                worst_mgf_ratio=max(worst_mgf_ratio,float(mgf.max()/math.exp(.5)))
                check(bool(np.all(mgf<=math.exp(.5)+TOL)),'pointwise Rademacher quadratic mgf')
                check(dminus.mean()<=V+TOL,'reverse expectation transfer')
                check(np.exp(dminus/(4*V)).mean()<=math.exp(.5)+TOL,'reverse exponential transfer')
                if V<.25:
                    forward_cases+=1
                    check(np.exp((lam-1)*dplus).mean()<=math.exp(.5)+TOL,'forward exponential transfer')
                    check(dplus.mean()<=2*V/(1-4*V)+TOL,'forward expectation transfer')
                cases+=1
    check(max_chain_gap<1e-11,'KL chain rule')
    return dict(abstract_transfer_instances=cases,small_risk_forward_instances=forward_cases,
        max_kl_chain_absolute_error=max_chain_gap,max_mgf_over_proved_cap=worst_mgf_ratio)


def norm(a): return float(np.linalg.norm(a,2))

def softmax(z):
    z=np.asarray(z);return np.exp(z-logsumexp(z))


def decoder_checks() -> dict:
    """No-training, standard two-layer causal residual decoder fixture.

Checks the new reference-query hybrid bound on all short token paths while
higher-layer suffix K/V are generated separately in both models.
"""
    max_violation=-math.inf;max_suffix_gap=0.;max_chain_gap=0.;paths=0;models=0
    for modelseed in [71,73,79,83]:
        rg=np.random.default_rng(modelseed);d=3;L=2;n=4;m=2;T=5
        embedding=rg.normal(size=(2,d))*.25
        layers=[];radius=max(np.linalg.norm(x) for x in embedding);const=[]
        for l in range(L):
            Q,K,V,O=[rg.normal(size=(d,d))*.20 for _ in range(4)]
            layers.append((Q,K,V,O))
            wq,wk,wv,so=[norm(x) for x in (Q,K,V,O)]
            r=max(wq,wk)*radius; vr=wv*radius;B=r*r
            a=1+2*so*vr*r*wq;b=so*math.exp(2*B)*(wv+2*vr*r*wk)
            c=math.sqrt(2)*so*vr*math.exp(B)
            const.append((a,b,c,vr,B))
            radius+=so*vr
        head=rg.normal(size=(2,d))*.6
        def empty(): return [(np.zeros((0,d)),np.zeros((0,d)),np.zeros(0)) for _ in range(L)]
        def clone(cache): return [(k.copy(),v.copy(),w.copy()) for k,v,w in cache]
        def step(token,cache):
            h=embedding[token].copy();states=[h.copy()];queries=[]
            new=[]
            for (Q,K,V,O),(keys,values,weights) in zip(layers,cache):
                query=Q@h;keys=np.vstack([keys,K@h]);values=np.vstack([values,V@h]);weights=np.r_[weights,1.]
                att=softmax(keys@query+np.log(weights))@values
                queries.append(query);h=h+O@att;states.append(h.copy());new.append((keys,values,weights))
            return head@h,new,states,queries
        prefix=empty()
        for tok in [0,1,1,0]: _,prefix,_,_=step(tok,prefix)
        for cseed in range(4):
            crg=np.random.default_rng(1000+10*modelseed+cseed);summary=[]
            for keys,values,weights in prefix:
                ix=crg.choice(n,m,replace=False);summary.append((keys[ix].copy(),values[ix].copy(),np.full(m,n/m)))
            logps=[];logqs=[];pathfw=[];pathrv=[]
            for ys in itertools.product([0,1],repeat=T):
                pc=clone(prefix);qc=clone(summary);errors=np.zeros((L+1,T))
                lp=lq=0.;locfw=locrv=0.;previous=1
                for t,y in enumerate(ys,1):
                    zp,pc,ps,pquery=step(previous,pc);zq,qc,qs,_=step(previous,qc)
                    for l in range(L+1): errors[l,t-1]=np.linalg.norm(qs[l]-ps[l])
                    for l,(a,b,c,vr,B) in enumerate(const):
                        q=pquery[l];kp,vp,_=prefix[l];ks,vs,ws=summary[l]
                        dz=(np.exp(ks@q)@ws-np.exp(kp@q).sum())/n
                        dn=((np.exp(ks@q)*ws)@vs-np.exp(kp@q)@vp)/(n*vr)
                        forcing=c*n/(n+t)*math.sqrt(dz*dz+float(dn@dn))
                        rhs=a*errors[l,t-1]+b*errors[l,:t].sum()/(n+t)+forcing
                        violation=errors[l+1,t-1]-rhs;max_violation=max(max_violation,violation)
                        check(violation<=TOL,'reference-query hybrid recurrence')
                        if l>0:
                            max_suffix_gap=max(max_suffix_gap,float(np.max(np.abs(qc[l][1][m:]-pc[l][1][n:]))))
                    pp=softmax(zp);qq=softmax(zq)
                    locfw+=float(qq@(np.log(qq)-np.log(pp)));locrv+=float(pp@(np.log(pp)-np.log(qq)))
                    lp+=math.log(pp[y]);lq+=math.log(qq[y]);previous=y
                logps.append(lp);logqs.append(lq);pathfw.append(locfw);pathrv.append(locrv);paths+=1
            logps=np.array(logps);logqs=np.array(logqs);p=np.exp(logps);q=np.exp(logqs)
            check(abs(p.sum()-1)<TOL and abs(q.sum()-1)<TOL,'decoder law normalization')
            max_chain_gap=max(max_chain_gap,abs(q@(logqs-logps)-q@pathfw),abs(p@(logps-logqs)-p@pathrv))
        models+=1
    check(max_suffix_gap>1e-8,'fixture must actually change upper-layer suffix KV')
    check(max_chain_gap<1e-11,'decoder chain rule')
    return dict(untrained_two_layer_fixtures=models,compression_seeds_per_fixture=4,
        exact_token_paths=paths,max_recurrence_violation=max_violation,
        max_upper_layer_suffix_value_gap=max_suffix_gap,max_kl_chain_absolute_error=max_chain_gap)


def hardy_checks() -> dict:
    max_ratio=0.
    for _ in range(100):
        n=int(RNG.integers(1,128));T=int(RNG.integers(1,192));u=RNG.normal(size=T)
        hu=np.cumsum(u)/(n+np.arange(1,T+1))
        ratio=np.linalg.norm(hu)/np.linalg.norm(u);max_ratio=max(max_ratio,ratio)
        check(ratio<=2+TOL,'Hardy norm')
    return dict(random_hardy_checks=100,max_observed_norm_ratio=max_ratio)


def main():
    results=dict(status='ALL_NUMERICAL_CHECKS_PASSED',
      scope='Proof/implementation sanity checks only; no training, no measured GS-Thin performance, no novelty certification',
      analytic=analytic_checks(),abstract_transfer=tree_transfer_checks(),
      causal_decoder=decoder_checks(),hardy=hardy_checks())
    (ROOT/'verification_results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
