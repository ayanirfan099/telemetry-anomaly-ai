"""Plot the fixed demo policy using bundled synthetic records, without tuning it."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import TelemetryModel, read_csv  # noqa: E402
from generate_data import FEATURES  # noqa: E402

model = TelemetryModel()
rows, values = read_csv(ROOT / 'data/test.csv')
truth = np.array([int(row['is_anomaly']) for row in rows])
scores = -model.model.score_samples(values)
report = model.evaluate()
out = ROOT / 'reports'
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
for label, text in [(0, 'Normal'), (1, 'Artificial anomaly')]:
    axes[0].hist(scores[truth == label], bins=24, alpha=.6, label=text)
axes[0].axvline(model.threshold, color='black', linestyle='--', label='Calibration threshold')
axes[0].set(title='Fixed-policy score distribution', xlabel='Anomaly score (not probability)', ylabel='Records')
axes[0].legend(fontsize=7)
axes[1].bar(['Precision', 'Recall', 'False-positive rate'],
            [report['precision'], report['recall'], report['false_positive_rate']], color=['#267f87', '#c67b45', '#667e95'])
axes[1].set(title='Operating point selected on normal calibration data', ylim=(0, 1), ylabel='Fraction')
axes[1].tick_params(axis='x', labelsize=8)
# Batch size is capped at 500 by the public API; both nonempty halves meet the PSI minimum.
records = [{feature: float(row[feature]) for feature in FEATURES} for row in rows]
drift = [model.predict_batch(batch)['population_stability_index'] for batch in [records[:300], records[300:]]]
x = np.arange(len(FEATURES))
for offset, d, label in [(-.2, drift[0], 'Test rows 1–300'), (.2, drift[1], 'Test rows 301–600')]:
    axes[2].bar(x + offset, [d[f] for f in FEATURES], width=.4, label=label)
axes[2].set_xticks(x, ['CPU', 'Latency', 'Error rate', 'Requests'], rotation=20)
axes[2].set(title='PSI relative to training distribution', ylabel='PSI (descriptive, no universal cutoff)')
axes[2].legend(fontsize=7)
fig.suptitle('Synthetic service telemetry — diagnostic plots, not production performance')
fig.savefig(out / 'diagnostics.png', dpi=150)
plt.close(fig)
print('Saved reports/diagnostics.png; policy unchanged:', report['threshold_policy'])
