"""Verify A6 simulation truth, saved estimates and interval coverage; no PDF required."""
from pathlib import Path
import datetime
import hashlib
import json
import re

import nbformat
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contrast_check(data, extra=(), outcome='Y', group=None):
    # Independent normal-equation / influence-vector check, without importing analysis.py.
    x = np.column_stack([np.ones(len(data)), data.A, data.X, data.G, data.A * data.G]
                        + [data[name] for name in extra])
    target = data[outcome].to_numpy()
    inverse = np.linalg.inv(x.T @ x)
    coefficient = np.linalg.solve(x.T @ x, x.T @ target)
    contrast = np.zeros(x.shape[1])
    contrast[1] = 1
    contrast[4] = data.G.mean() if group is None else group
    residual = target - x @ coefficient
    leverage = np.sum((x @ inverse) * x, axis=1)
    influence = (x @ (inverse @ contrast)) * residual / (1 - leverage)
    return float(contrast @ coefficient), float(np.sqrt(np.sum(influence ** 2)))


def main():
    nb = nbformat.read(BASE / 'A6.ipynb', as_version=4)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    assert not [o for c in code for o in c.outputs if o.output_type == 'error']
    assert nb.cells[0].source.startswith('# Abstract')
    stream = '\n'.join(o.get('text', '') for c in code for o in c.outputs if o.output_type == 'stream')
    paths = re.findall(r'RESULT_DIR:\s*(results/\S+)', stream)
    assert paths, 'Save the completed notebook before running verification.'
    run = (BASE / paths[-1]).resolve()
    assert run.is_relative_to((BASE / 'results').resolve())
    complete = read_json(run / 'completed.json')
    meta = read_json(run / 'run_metadata.json')
    code_hash = hashlib.sha256(''.join(c.source for c in code).encode()).hexdigest()
    assert code_hash == complete['notebook_code_sha256'] == meta['notebook_code_sha256']
    assert sha(BASE / 'analysis.py') == complete['analysis_source_sha256'] == meta['analysis_source_sha256']
    assert sha(run / 'protocol.json') == complete['protocol_sha256'] == meta['protocol_sha256']
    protocol = read_json(run / 'protocol.json')
    assert protocol['assignment'] == 6 and protocol['seed'] == meta['seed']
    assert sha(BASE / 'data/raw/manifest.json') == complete['manifest_sha256'] == meta['manifest_sha256']
    manifest = read_json(BASE / 'data/raw/manifest.json')
    for name, expected in manifest['files'].items():
        assert sha(BASE / 'data/raw' / name) == expected, name

    observed = pd.read_csv(BASE / 'data/raw/observed.csv')
    oracle = pd.read_csv(BASE / 'data/raw/oracle.csv')
    assert len(observed) == len(oracle) == protocol['n'] == 2500
    assert set(observed) == {'X', 'G', 'A', 'M', 'C', 'Y', 'Ypre'}
    assert set(['U', 'true_propensity', 'Y0', 'Y1']).isdisjoint(observed.columns)
    assert np.allclose(observed, oracle[observed.columns])
    assert np.allclose(oracle.Y, np.where(oracle.A == 1, oracle.Y1, oracle.Y0))
    assert np.allclose(oracle.Y1 - oracle.Y0, 2.56 + oracle.G)
    truth = float((oracle.Y1 - oracle.Y0).mean())
    assert np.isclose(truth, 2 + 0.8 * 0.7 + oracle.G.mean())
    assert np.isclose(truth, 3.0664)
    facts = read_json(run / 'summary.json')
    assert facts['synthetic'] and facts['n'] == 2500
    assert np.isclose(facts['sample_ATE'], truth)
    assert np.isclose(facts['population_ATE'], 2 + 0.8 * 0.7 + 0.5)
    estimates = pd.read_csv(run / 'estimates.csv').set_index('method')
    specifications = {'Observed_XG': (), 'Oracle_XGU': ('U',),
                      'Observed_plus_mediator': ('M',), 'Oracle_plus_mediator': ('U', 'M'),
                      'Observed_plus_collider': ('C',)}
    for method, extra in specifications.items():
        data = oracle if 'U' in extra else observed
        value, se = contrast_check(data, extra)
        saved = estimates.loc[method]
        assert np.isclose(value, saved['estimate'], rtol=1e-10, atol=1e-10), method
        assert np.isclose(se, saved['se'], rtol=1e-10, atol=1e-10), method
        assert np.isclose(saved.lower, value - 1.96 * se)
        assert np.isclose(saved.upper, value + 1.96 * se)
        assert np.isclose(saved.difference_from_total_truth, value - truth)
    unadjusted = observed.loc[observed.A == 1, 'Y'].mean() - observed.loc[observed.A == 0, 'Y'].mean()
    assert np.isclose(unadjusted, estimates.loc['Unadjusted', 'estimate'])
    assert np.isclose(estimates.loc['Observed_XG', 'estimate'], facts['observed_estimate'])
    assert np.isclose(estimates.loc['Oracle_XGU', 'estimate'], facts['oracle_estimate'])
    negative = pd.read_csv(run / 'negative_control.csv').set_index('method')
    for method, extra in [('Observed_XG', ()), ('Oracle_XGU', ('U',))]:
        value, se = contrast_check(oracle if extra else observed, extra, outcome='Ypre')
        assert np.isclose(value, negative.loc[method, 'estimate'])
        assert np.isclose(se, negative.loc[method, 'se'])
    assert np.isclose(negative.loc['Observed_XG', 'estimate'], facts['negative_control_observed'])
    heterogeneity = pd.read_csv(run / 'heterogeneity.csv')
    for _, row in heterogeneity.iterrows():
        extra = ('U',) if row['method'] == 'Oracle_XGU' else ()
        value, se = contrast_check(oracle if extra else observed, extra, group=row['group'])
        assert np.isclose(value, row['estimate']) and np.isclose(se, row['se'])
        assert row.group_n == (observed.G == row['group']).sum()
        assert np.isclose(row.true_total_effect, 2.56 + row['group'])
        assert np.isclose(row.true_net_benefit_at_cost_2_8, row.true_total_effect - protocol['cost'])

    reps = pd.read_csv(run / 'sensitivity_replicates.csv')
    sensitivity = pd.read_csv(run / 'sensitivity_summary.csv')
    assert set(reps.gamma) == set(protocol['gamma_grid'])
    assert set(reps.method) == {'Observed_XG', 'Oracle_XGU'}
    assert len(reps) == 4 * 200 * 2 == 1600
    assert not reps.duplicated(['gamma', 'replicate', 'method']).any()
    assert np.allclose(reps.bias, reps['estimate'] - reps.truth)
    covered = (reps.lower <= reps.truth) & (reps.upper >= reps.truth)
    assert np.array_equal(covered, reps.covered)
    z = norm.ppf(0.975)
    for _, row in sensitivity.iterrows():
        part = reps[(reps.gamma == row.gamma) & (reps.method == row['method'])]
        assert len(part) == row.repetitions == protocol['simulation_repetitions'] == 200
        assert set(part.replicate) == set(range(200)) and (part.n == protocol['simulation_n']).all()
        n = len(part); probability = part.covered.mean()
        assert np.isclose(row.coverage, probability)
        assert np.isclose(row.mean_bias, part.bias.mean())
        assert np.isclose(row.bias_MCSE, part.bias.std(ddof=1) / np.sqrt(n))
        assert np.isclose(row.RMSE, np.sqrt(np.mean(part.bias ** 2)))
        assert np.isclose(row.coverage_MCSE, np.sqrt(probability * (1 - probability) / n))
        center = (probability + z*z / (2*n)) / (1 + z*z/n)
        half = z * np.sqrt(probability * (1-probability)/n + z*z/(4*n*n)) / (1 + z*z/n)
        assert np.isclose(row.coverage_Wilson_lower, max(0, center - half), atol=1e-12)
        assert np.isclose(row.coverage_Wilson_upper, min(1, center + half), atol=1e-12)
    gamma1 = reps[(reps.gamma == 1) & (reps.method == 'Observed_XG')]
    assert gamma1.covered.mean() == facts['observed_coverage_gamma_1'] == 0
    reflection = [c for c in nb.cells if c.get('id') == 'student-verification-reflection']
    assert len(reflection) == 1 and reflection[0].source.strip()
    histories = sorted((BASE / 'results').glob('*/completed.json'))
    for previous in histories:
        old = read_json(previous.parent / 'summary.json')
        for key in ['sample_ATE', 'observed_estimate', 'oracle_estimate', 'observed_coverage_gamma_1', 'negative_control_observed']:
            assert np.isclose(old[key], facts[key], rtol=1e-10, atol=1e-10), key
    record = {
        'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'verified_by': 'Codex automated checks; not student independent verification',
        'notebook': 'A6.ipynb', 'result_directory': str(run.relative_to(BASE)),
        'notebook_code_cells': len(code), 'notebook_code_sha256': code_hash,
        'analysis_source_sha256': complete['analysis_source_sha256'],
        'protocol_sha256': complete['protocol_sha256'],
        'raw_data_hashes_verified': len(manifest['files']), 'synthetic_rows': len(oracle),
        'independently_recomputed_sample_ATE': truth,
        'independent_regression_and_HC3_checks': len(specifications),
        'negative_control_and_group_contrasts_checked': True,
        'sensitivity_rows_checked': len(reps),
        'coverage_and_monte_carlo_summaries_checked': True,
        'historical_metric_runs_checked': len(histories),
        'author_reflection_retained': True, 'PDF_or_root_protocol_required': False,
    }
    (run / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print('VERIFIED', record['result_directory'])
    print('Executed code cells:', len(code), 'synthetic rows:', len(oracle))
    print('Sample ATE:', f'{truth:.7f}')
    print('Observed / oracle estimates:', f"{facts['observed_estimate']:.9f}", f"{facts['oracle_estimate']:.9f}")
    print('Independent regression/HC3 checks and all 1,600 sensitivity rows verified.')
    print('No PDF, exporter, Markdown report or root protocol is required.')


if __name__ == '__main__':
    main()
