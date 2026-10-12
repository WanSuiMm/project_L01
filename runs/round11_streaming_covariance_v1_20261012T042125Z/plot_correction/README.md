# Fisher figure correction

Use [fisher_predictions.png](fisher_predictions.png) or its PDF in this
directory. The frozen v1 figure at `../analysis/fisher_predictions.*`
multiplied the stored predictions by amplitude squared a second time.
At amplitude 0.25 this plotted y values 16 times too small; the amplitude-1
panel was unaffected. Scientific CSV values, relative errors, aggregates,
screen results and the other figures were correct and remain unchanged.

Current `experiments/analyze_round11.py` plots the already scaled CSV
prediction directly. The original run source copy and figures are preserved
for audit. These corrected figures were generated in a new directory from
the same frozen Fisher CSV; [manifest.json](manifest.json) binds the CSV,
corrected analyzer and figures by SHA-256. The release verifier checks the
original payload and corrected-figure hashes separately.

To regenerate the correction into a new directory from the repository root:

```text
python -B scripts/correct_round11_fisher_plot.py --input-dir runs/round11_streaming_covariance_v1_20261012T042125Z --output-dir runs/NEW_PLOT_CORRECTION
```
