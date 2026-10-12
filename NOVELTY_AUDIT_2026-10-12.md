# L01 novelty audit and remaining research questions

Audit date: 2026-10-12. Reviewed repository head: `7f59ab24f481cde4bf4cbd4d2644e303e96c2025`.
The subsequent [user-provided Round 14B protocol review](ROUND14B_PROTOCOL_REVIEW_2026-10-12.md)
adds the latest source checks and a concrete decision about the proposed final
transfer test. It prioritizes the existing Round 7 theorem candidate rather than
starting another broad learned-allocation project.
This is a bounded primary-source literature audit and research decision note.
It changes no frozen experiment, threshold, result or theorem status. No new
experiment was launched for this audit.

## Decision

L01 has a useful finite observation: compression-event interactions explain a
large temporal-policy contrast in the trained histogram setting. The current
evidence does not establish a broadly useful risk predictor or compressor.
Round 14 remains a failed transfer test. A positive replacement experiment
alone would not establish novelty or finish the paper.

The most useful next question is whether a compression-time score can make
better **finite-budget decisions about future generation**, rather than merely
explain an already evaluated trajectory. This is a candidate research question,
not a claim that existing literature has no related solutions.

## Closest verified work

| Primary source and version | Already covered | Boundary relevant to L01 |
|---|---|---|
| [Streaming Attention Approximation via Discrepancy Theory / BalanceKV](https://papers.neurips.cc/paper_files/paper/2025/hash/1de510838faddd72f79a7a0e9c564f0c-Abstract-Conference.html), NeurIPS 2025; algorithm and Theorem 3.1 | Geometry-aware dependent selection and streaming attention approximation. | Balanced or correlated sampling is established. Whole-generator sequence-risk control is a different objective, but changing the objective alone is not an algorithmic contribution. |
| [Behavior-Preserving KV Cache Compression](https://arxiv.org/pdf/2610.06479), v1, 2026-10-05; Sections 3.1, 4.2–4.4, Appendix B and Limitations | Output-distribution KL scoring, approximate residual propagation, joint-mask marginal refinement, and future-position evaluation on a reference continuation. | The authors explicitly limit the compression-time signal to available queries and give no guarantee for arbitrary unseen continuations. Joint eviction effects and output KL are not new L01 contributions. |
| [LookaheadKV](https://arxiv.org/html/2603.10899v1), v1, 2026-03-11; Sections 2 and 3.2 | Learned lookahead tokens and LoRA predict the attention importance of the full-model future response without explicit draft generation. | Future-aware importance prediction is established. Its attention target is distinct from the finite sequence risk of each candidate compressed model generating its own continuation. This distinction still needs an actionable method or structural result. |
| [The risk of KV cache compression](https://arxiv.org/html/2607.01520v1), v1, 2026-07-01; Sections 3–5, Theorem 5.4 | Query-weighted response covariance, minimax attention risk, query-aware/agnostic separation, and causal merge-reduce lifting. | Neither covariance geometry nor causal streaming is an empty niche. Its fixed query-distribution attention objective is distinct from a full multilayer generator whose query distribution changes with compression. |
| [KV Cache Compression Through the Lens of Transform Coding](https://arxiv.org/html/2608.14191v1), v1, 2026-08-14; Assumption 1, Theorem 1, Section VII | Attention-aware bit allocation and distortion decomposition under independent, zero-mean quantization errors. | The paper explicitly identifies cross-token independence as an idealization and weaker dependence assumptions as future work. A covariance expansion by itself is elementary; a useful extension must change an implementable allocation or compression decision. |
| [KVarN](https://arxiv.org/abs/2606.03458), v1, 2026-06-02; Sections 3.2 and 3.4 | Pseudo-decode evaluation and error accumulation across layers and decoding steps for KV quantization. | General statements that autoregressive cache errors accumulate are established. L01 needs a more specific mechanism, condition or decision benefit. |
| [DistillCache](https://arxiv.org/html/2608.08878v1), v1, 2026-08-09; Sections 3.1–3.3 and 5 | Sequential learned eviction, separate teacher/student caches, per-step output-KL rewards and discounted policy optimization. | A KL-trained sequential cache policy is established. Its training history law must be established before equating its reward to a whole-sequence KL. The paper explicitly notes independent token scoring and training/serving costs. |
| [ForesightKV](https://arxiv.org/html/2602.03203v1), v1, 2026-02-03; Section 3.3 | Future-attention supervision followed by RL to address eviction-induced state shift; sequence rewards over a supplied reasoning trace. | State-shift awareness and long-term cache contribution are established. The written training objective varies eviction traces for a given token sequence; this does not itself identify the law of freely generated continuations. |
| [WitCert](https://arxiv.org/html/2607.28699v3), v3, 2026-08-16; Sections 4.2, 6.4 and 6.4.6 | Runtime attention-TV witnesses, an adaptive-safe deterministic tier, a non-adaptive probabilistic tier, and repair gating. | Runtime detection and gating are established topics. The paper states that online eviction witnesses and composed multilayer certificates remain open in its framework. Its local TV target is not a full-generator sequence-KL certificate. This audit has not independently checked its proofs or artifacts. |
| [Error Certificates for KV-Cache Eviction via Randomized Design](https://arxiv.org/html/2607.21475v3), v3, 2026-08-29; Sections 3 and 8 | Poisson-tail retention, inclusion-probability correction, retained-set attention-error inference, and empirical attribution/repair experiments. | Online randomized eviction certificates are already studied. The discussion keeps task-level meaning empirical and identifies incomplete anytime-extension issues. A general claim that certifying eviction is untouched would be wrong. |
| [Risk-Controlled KV-Cache Eviction](https://arxiv.org/abs/2609.27981), v1, 2026-09-23; abstract-level scope check | Calibration-based certification of population task-degradation frequency and policy retention selection, with full-cache fallback. | Population risk under a calibration contract differs from per-request model-law KL; statistical certification of eviction is nonetheless established prior work. |

[HeadQ](https://arxiv.org/abs/2605.03562) is withdrawn: the current arXiv record
identifies v3 on 2026-05-19 as withdrawn by the author because of ethical
concerns. Earlier accessible versions or the abstract must not be presented
as verified empirical support. This audit excludes its results from the
established-evidence comparison. Withdrawal does not establish that every
mathematical idea in an earlier version is false or novel elsewhere.

This audit does not independently validate every proof or reproduce these
papers. No mathematical error in them has been established here. An explicit
assumption or limitation is not a contradiction in the paper's stated result.

## What L01 currently contributes

The [Round 13 record](RESULTS.md) shows that full event JVPs through a causal
unroll, combined with categorical output Fisher cross terms on the same saved
Q-generated histories, recover a large Block/Alternating contrast in five
two-layer histogram models. Actual KL is 1.216538/0.0997674; full quadratic is
1.220090/0.0997077; diagonal-only is 0.178589/0.181480.

That quadratic is retrospective and averaged over endpoint-Q histories. It is
not the P-law sequence-Fisher Taylor coefficient, a heldout predictor, a
certificate, or a deployable policy. The Fisher identity and existence of
off-diagonal terms are standard. The restricted Round 10 bounded/logarithmic/
linear constructions remain candidates for a narrow formal comparison; this
audit does not establish that their joint statement is unprecedented.

Round 14 qualifies only 4/5 seeds and fails both frozen policy-level Fisher
transfer screens. Restricting attention to the four qualified seeds still
fails. Its old retrieval table remains exact, only generated suffix pairs are
thinned, and local-error comparability passes only 1/5 seeds. Thus it does not
isolate temporal order or test compression of essential retrieval memory.
The small KL on qualified seeds also causes no observed answer failure; a
relative prediction error alone does not establish a material practical harm.

## Ranked questions worth testing

### 1. When does a compression-time score select the wrong future cache?

**Established limitation:** Behavior-Preserving scoring cannot observe all
future queries. LookaheadKV predicts future attention; DistillCache already
trains sequential policies with KL rewards, and ForesightKV explicitly addresses
eviction-induced state shift. Thus neither future awareness, RL, nor state-shift
awareness is the additional contribution. The candidate question is a verifiable,
inexpensive condition for when their scores preserve the ranking of finite
whole-generator risks, and a useful correction when that condition fails.
This is our inference from their objectives, not a demonstrated defect or a
globally established absence of such conditions in other work.

For a fixed physical budget, compare a current-query score with
`R_H(a) = KL(Q^a_{1:H} || P_{1:H})`, where every candidate `a` has its own
generated-history law. Evaluate both KL directions as sensitivity checks;
do not attribute a direction mismatch to a horizon or history-law effect.
The chain rule requires matching laws: reverse sequence KL uses Q histories
with conditional KL(Q||P), while forward sequence KL uses P histories with
conditional KL(P||Q). A per-step KL reward, discounting, or a fixed token trace
does not automatically satisfy either identity. The identities themselves are
standard and not a new result.

A useful contribution must improve heldout selection at finite compression,
or give a verifiable condition under which the cheaper score is sufficient.
Merely observing a correlation on the evaluated paths is insufficient. Charge
the time and memory used by lookahead, sketches, calibration and extra passes.
An oracle future score is an evaluation ceiling, not a deployable method.

**Smallest decision test:** reuse frozen models, hold out tables and masks,
compress memory needed by the task, and compare actual mask-selection regret
and task loss at a fixed record budget. Separate current-query approximation,
future-query coverage and closed-loop history effects rather than bundle them.
Include relevant learned and joint-mask baselines before a method claim; a toy
implementation of a score must not be reported as a reproduction of a full
published system. A mask-ranking pilot can precede that larger comparison.
Do not claim novelty from a toy timing at which the next token is an independent
random query: its current score is uninformative by construction.

**Stop criterion:** if current/future-attention baselines already select equally
good caches, or improvements disappear after accounting for overhead, do not
extend this into a broader method claim.

### 2. Can controlling error dependence improve generation at the same cost?

**Established limitation:** Transform Coding's independence assumption leaves
correlated errors outside its distortion decomposition. BalanceKV already
uses dependence beneficially. The remaining candidate question is a causal,
generation-sensitive coupling of actual compression errors, using only
information available at the decision time.

The necessary endpoint is a better finite-budget generation/quality tradeoff
than independent errors and geometry-aware balancing. Preserve marginal
quantizer/selector laws where the protocol claims a dependence-only contrast,
and include residual state in the memory budget. Match local error empirically
without selecting seeds or test cases after seeing the result.

**Smallest decision test:** one frozen task in which the compressed memory is
actually used, a fixed budget, independent versus a predeclared causal feedback
policy, and all original model seeds. A fixed Alternating schedule passing on
a histogram task does not itself satisfy this requirement.

**Stop criterion:** if gains require future trajectories, negligible perturbation
amplitudes, uncounted state, or a favorable token-role selector, retain the
diagnostic and stop the algorithmic claim.

### 3. Can we detect when a local quadratic is unsafe at physical compression?

**L01 observation:** Round 13 fits well, while Round 14 does not pass the same
relative-error screen. This motivates an applicability question; it does not
prove that Taylor truncation uniquely caused the difference.

The useful target would be an inexpensive warning or ranking correction that
works on heldout masks at the actual compression strength. Exact path-integral
identities or shrinking the strength until Taylor expansion fits are not such
a contribution. Neither is re-demonstrating joint-mask nonadditivity, already
addressed by Behavior-Preserving compression. WitCert also prevents a broad
claim that runtime error witnesses or repair gating are unexplored.

**Bounded diagnostic option:** eight fixed prefix-row merge axes admit 256
physical subset interventions. Exact finite differences can distinguish
additive, pairwise and higher-order answer-NLL effects. Use value-swapped
controls and disjoint calibration/evaluation tables. This is a conditional
answer experiment; it does not establish whole-generation sequence risk.

**Stop criterion:** if a warning has no benefit over simple confidence/margin
and joint-mask baselines, or succeeds only at tiny strength, do not launch a
larger sweep to rescue it.

## Recommendation and experiment status

Prioritize a bounded decision test for question 1, with question 3 as a diagnostic
before investing in a new predictor. Further source checks substantially narrow
the opportunity: an RL allocator with a KL reward is already prior work. Question
2 is a plausible separate algorithmic project but is less suitable for quickly
closing L01. None is certified novel by this bounded search, and none warrants
calling a successful replacement for Round 14 a finished paper by itself.

A corrected essential-memory pilot may be worth doing. It must have a new
protocol and run identity; it cannot replace Round 14 or silently redefine its
frozen success criterion. No such pilot has been implemented or launched by
this audit. Preserve the existing positive histogram evidence and negative
transfer evidence regardless of the next result.
