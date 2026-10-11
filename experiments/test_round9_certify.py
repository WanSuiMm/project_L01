"""Fast tests for mathematical bounding subroutines and full-tree coverage."""
import unittest, math, sys
import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from round9_certify import (demo_model,run_cert,layernorm_chord_lip,
                            softmax_perturb_bound,norm,TinyCausalDecoder)

torch.set_default_dtype(torch.float64)

class TestRound9(unittest.TestCase):
    def test_layernorm_local_radius(self):
        torch.manual_seed(17)
        ln=torch.nn.LayerNorm(8).double()
        with torch.no_grad(): ln.weight.copy_(torch.tensor([1.1,.5,1.4,.8,.7,1.2,1.,.6]))
        for _ in range(300):
            x=torch.randn(8)
            radius=float(torch.rand(())*.2)
            u=torch.randn(8);u=u/(norm(u)+1e-20)*radius
            lhs=norm(ln(x+u)-ln(x))
            rhs=layernorm_chord_lip(ln,x,radius)*radius
            self.assertLessEqual(lhs,rhs+1e-10)

    def test_score_gauge_zero(self):
        k=torch.ones((8,4),dtype=torch.float64)*2.
        v=torch.randn((8,3),dtype=torch.float64)
        p=torch.ones(8,dtype=torch.float64)/8
        A=(p[:,None]*v).sum(0)
        bound,_=softmax_perturb_bound(k,v,torch.ones(4),p,A,
              torch.zeros(8),torch.zeros(8),zeta=5.,global_value_bound=20.)
        self.assertLessEqual(abs(bound),1e-12)

    def test_probability_bounds_for_perturbed_softmax(self):
        torch.manual_seed(18)
        for _ in range(200):
            n=8;d=5;dv=4
            k=torch.randn(n,d)*.5;v=torch.randn(n,dv);q=torch.randn(d)
            weights=torch.rand(n)+.1
            p=torch.softmax(k@q+weights.log(),0);a=(p[:,None]*v).sum(0)
            e_q=torch.rand(())*.2; e_k=torch.rand(n)*.15; e_v=torch.rand(n)*.12
            deltaq=torch.randn(d);deltaq=deltaq/(norm(deltaq)+1e-10)*float(e_q)
            deltak=torch.randn(n,d);deltak=deltak/(torch.linalg.vector_norm(deltak,dim=1)[:,None]+1e-10)*e_k[:,None]
            deltav=torch.randn(n,dv);deltav=deltav/(torch.linalg.vector_norm(deltav,dim=1)[:,None]+1e-10)*e_v[:,None]
            pq=torch.softmax((k+deltak)@(q+deltaq)+weights.log(),0)
            aq=(pq[:,None]*(v+deltav)).sum(0)
            bound,_=softmax_perturb_bound(k,v,q,p,a,e_k,e_v,float(e_q),global_value_bound=100.)
            self.assertLessEqual(norm(aq-a),bound+1e-9)

    def test_full_3ary_histories_and_tree_kl(self):
        model=demo_model();model._source_kind='untrained_demo'
        prefix=np.random.default_rng(12).integers(0,2,size=12).tolist()
        for m,kind in ((11,'closest_pair'),(6,'greedy_pairs')):
            r,tree=run_cert(model,prefix,2,3,m,kind,include_exact_q=True)
            self.assertEqual(r['histories'],13)   # 1+3+9, not 1+2+4
            self.assertEqual(r['full_sequence_leaves'],27)
            self.assertLessEqual(r['exact_KL_QP'],r['cert_QP_upper']+1e-10)
            self.assertLessEqual(r['exact_KL_PQ'],r['cert_PQ_upper']+1e-10)
            # No Q computations are used to construct the bound.
            r2,_=run_cert(model,prefix,2,3,m,kind,include_exact_q=False)
            self.assertAlmostEqual(r['cert_QP_upper'],r2['cert_QP_upper'])
            self.assertAlmostEqual(r['cert_PQ_upper'],r2['cert_PQ_upper'])

if __name__=='__main__':unittest.main()
