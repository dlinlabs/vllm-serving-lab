"""Draw the grey-box forecast figure from the archived report (matplotlib only)."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = (('ridge', 'PR #26 ridge (service_no_plan)'), ('nowcast', 'Trailing-60 s nowcast'),
          ('simulator', 'Grey-box simulator'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=HERE / 'full-report.json.gz')
    parser.add_argument('--output', type=Path, default=HERE / 'greybox-forecast.png')
    args = parser.parse_args()
    report = json.loads(gzip.decompress(args.report.read_bytes()))
    scores = report['scores']
    runs = list(scores['ridge']['per_run_mae'])
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2))
    width = .27
    for k, (model, label) in enumerate(MODELS):
        values = [scores[model]['per_run_mae'][r] for r in runs] + [scores[model]['mae']]
        left.bar([i + (k - 1) * width for i in range(len(values))], values, width,
                 label=f"{label}: {scores[model]['mae']:.3f}")
    left.set_xticks(range(len(runs) + 1), [r.replace('cap-range-v1-', '').replace('-client', '') for r in runs] + ['all'],
                    rotation=35, ha='right')
    left.set_ylabel('5 s goodput MAE (req/s)')
    left.set_ylim(0, 1.3 * max(max(scores[m]['per_run_mae'].values()) for m, _ in MODELS))
    left.set_title(f"Chronological test origins (t ≥ 200 s, n = {report['protocol']['test_origins']})")
    left.legend(fontsize=8)

    curves = {c['held']: c['rows'] for c in report['cap_curves']}
    caps = [r['cap'] for r in curves[None]]
    right.plot(caps, [r['goodput'] for r in curves[None]], 'o-', color='C2', label='Predicted goodput')
    right.plot(caps, [r['admitted'] for r in curves[None]], 's--', color='C0', label='Predicted admitted')
    measured = report['measured_fixed_caps']
    right.scatter([m['cap'] for m in measured], [m['goodput'] for m in measured], marker='x', s=60, color='C2',
                  label='Measured goodput')
    right.scatter([m['cap'] for m in measured], [m['admitted'] for m in measured], marker='+', s=80, color='C0',
                  label='Measured admitted')
    held = [(h, next(r for r in rows if r['cap'] == h)) for h, rows in curves.items() if h is not None]
    right.scatter([h for h, _ in held], [r['goodput'] for _, r in held], facecolors='none', edgecolors='C3', s=80,
                  label='Goodput with that cap held out of the fit')
    right.set_xticks(caps)
    right.set_xlabel('Fixed cost cap')
    right.set_ylabel('Rate over a 300 s run (req/s)')
    right.set_title('Open-loop cap curve from an idle start (reference arrivals: fixed-32-r1)')
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(args.output)


if __name__ == '__main__':
    main()
