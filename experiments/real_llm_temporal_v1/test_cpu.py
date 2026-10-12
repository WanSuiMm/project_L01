import json
import math
from pathlib import Path
import tempfile
import unittest
from run_qwen_probe import cpu_math_self_test, event_count, signs_for, summarize

class TestTemporalProtocol(unittest.TestCase):
    def test_mask_semantics(self):
        cpu_math_self_test()
    def test_event_count_and_signs(self):
        self.assertEqual(event_count(130),64)
        for p in ('block','alternating','balanced_random'):
            a=signs_for(p,1,64,random_seed=7)
            b=signs_for(p,-1,64,random_seed=7)
            self.assertEqual(sum(a),0)
            self.assertEqual(sum(b),0)
            self.assertEqual(a,[-x for x in b])
        self.assertEqual(sum(signs_for('block',1,63)),-1)
    def test_summary_nesting(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            data=[]
            for prompt in ('p1','p2','p3'):
                for policy in ('block','alternating','balanced_random'):
                    for rep in (0,1):
                        for phase in (-1,1):
                            data.append(dict(prompt_id=prompt,category='reasoning',policy=policy,
                                             phase=phase,replicate=rep,horizon=8,events=3,eos_count=0,
                                             full_kl_q_to_p={'block':3.,'alternating':1.,'balanced_random':2.}[policy],
                                             local_attention_error_sq_mean=.01))
            path=Path(directory)/'rows.jsonl'
            path.write_text('\n'.join(json.dumps(x) for x in data)+'\n')
            (Path(directory)/'manifest.json').write_text(json.dumps({
                'status':'COMPLETE','expected_prompt_ids':['p1','p2','p3'],
                'expected_policy_ids':['block','alternating','balanced_random'],
                'expected_phases':[1,-1],'expected_replicate_ids':[0,1],
                'prompt_count':3,'paths_per_phase':2,'horizon':8,'event_count':3,
                'source_sha256':{'prompts':'fixture','runner':'fixture'},
                'provenance':{'model_reference':'tiny-fixture'},
                'completed_rows':len(data),
            }))
            s=summarize(directory)
            self.assertEqual(s['n_prompts'],3)
            self.assertEqual(s['n_positive_block_over_alt'],3)
            self.assertEqual(s['n_local_comparable_le1_25'],3)
            self.assertAlmostEqual(s['policy_aggregate']['block']['mean_kl'],3.)
            self.assertEqual(s['paired_prompt_bootstrap95'],[2.,2.])

    def test_summary_rejects_incomplete_grid(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            data=[]
            for policy in ('block','alternating','balanced_random'):
                for rep in (0,1):
                    for phase in (-1,1):
                        data.append(dict(prompt_id='p1',category='reasoning',policy=policy,
                                         phase=phase,replicate=rep,horizon=8,events=3,eos_count=0,
                                         full_kl_q_to_p=1.,local_attention_error_sq_mean=.01))
            data.pop()
            (Path(directory)/'rows.jsonl').write_text('\n'.join(json.dumps(x) for x in data)+'\n')
            (Path(directory)/'manifest.json').write_text(json.dumps({
                'status':'COMPLETE','expected_prompt_ids':['p1'],
                'expected_policy_ids':['block','alternating','balanced_random'],
                'expected_phases':[1,-1],'expected_replicate_ids':[0,1],
                'prompt_count':1,'paths_per_phase':2,'horizon':8,'event_count':3,
                'source_sha256':{'prompts':'fixture','runner':'fixture'},
                'provenance':{'model_reference':'tiny-fixture'},
                'completed_rows':12,
            }))
            with self.assertRaisesRegex(ValueError,'incomplete or duplicate manifest grid'):
                summarize(directory)

if __name__=='__main__':unittest.main()
