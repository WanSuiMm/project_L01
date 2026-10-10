# L01 Round 7 — law transfer and restricted thinning minimax

本目录是独立研究草案，不修改 `project_L01` 的任何冻结结果，也没有上传 GitHub。

## 阅读顺序

1. `THEOREM_AND_PROOF.md`：完整假设、概率转移证明、现有 GS-Thin 的实例化、等权类别下界。
2. `NOVELTY_AUDIT.md`：已核对前作、真正区别与未解决问题。
3. `verification_results.json`：实际执行的数值检查结果。
4. `verify_round7.py`：可复算的检查代码；两个 CSV 是解析/动态规划结果。

核心候选结论：记 `h=nT/(n+T)`。一次性每层 m 条等权原记录摘要，在统一模型范数和文献 GS-Thin 的次高斯接口下，`E_R KL(P||Q_R)<=C h/m²`；`C h/m²<1/4` 时有相应 `E_R KL(Q_R||P)` 与高概率上界。该等权类别中有 `Omega(h/m²)` 下界。任意正权/任意编码器的下界未解决。

## 重现

需要 Python 3、NumPy 和 SciPy：

```bash
python verify_round7.py
```

脚本执行：600 个 Bernoulli KL 不等式检查、653 个可行整数摘要计数检查、100 个等权 minimax 网格配置、100 个均匀抽样精确风险配置、36 个抽象 entropy-transfer 实例（31 个满足正向小风险条件）、4 个未训练双层 residual softmax decoder 的 16 个摘要实例共 512 条完整二元续写，以及 100 个 Hardy 随机数值检查。

这些是推导和实现的 sanity checks，不是训练实验，不是 GS-Thin 的性能评测，不是形式化证明或原创性证明。
