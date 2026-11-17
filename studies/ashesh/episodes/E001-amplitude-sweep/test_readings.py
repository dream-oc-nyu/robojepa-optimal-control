"""CPU-only contract tests; run with python test_readings.py."""
import numpy as np


def test_cost():
    from readings import cost
    # Float64 subtraction must happen BEFORE reduction, including float32 inputs.
    z = np.array([[1, -3], [5, 7]], dtype=np.float32)
    assert cost(z, np.ones_like(z)) == 3.5


def toy():
    # D = [4,4,1,2,3,0] mm. Real picks b; cmd picks c; world picks a.
    names = ['n0', 'n1', 'a', 'b', 'c', 'g']
    real = np.array([4, 4, 3, 1, 2, 0], np.float32)[:, None, None, None, None]
    cmd = np.array([8, 8, 5, 6, 4, 0], np.float32)[:, None, None, None, None]
    return dict(names=np.array(names), kind=np.array(['neutral', 'neutral', 'onray', 'onray', 'offray', 'goal']),
                commanded_xyz=np.array([[0, 0, 0], [0, 0, 0], [.003, 0, 0], [.002, 0, 0], [0, .001, 0], [.004, 0, 0]], np.float64),
                executed_delta14=np.zeros((6, 14), np.float64),
                ee_world=np.array([[0, 0, 0], [0, 0, 0], [.003, 0, 0], [.002, 0, 0], [.001, 0, 0], [.004, 0, 0]], np.float64),
                sim_seconds=np.ones(6, np.float64), rewritten=np.zeros(6, bool),
                z_real=np.repeat(real, 2, axis=1), z_pred_cmd=np.repeat(cmd, 2, axis=1),
                z_pred_exec=np.repeat(real, 2, axis=1), images=np.zeros((6, 2, 240, 320, 3), np.uint8))


def test_readings_contract():
    from readings import concordance, evaluate, validate_instance
    c = concordance([0, 0, 2, 3], [0, 1, 1, 3])
    assert c == dict(concordant=4, discordant=0, total=6, cost_ties=1,
                     distance_ties=1, both_ties=0, fraction=4/6)
    assert concordance([0, 0], [0, 0])['both_ties'] == 1
    r = evaluate(toy(), 'toy')
    assert r['regret_split']['predictor_interface_mm'] == 1
    assert r['regret_split']['representation_cost_mm'] == 1
    assert r['rankings']['all']['predicted-cmd']['selected'] == 'c'
    assert np.isclose(r['rankings']['all']['predicted-cmd']['progress_captured'], 1/3)
    assert r['rankings']['offray']['realized']['concordance']['fraction'] is None
    n = r['candidates'][0]
    assert n['ratio'] is None and n['G_cmd'] is None and n['cos_cmd_real'] is None
    assert r['candidates'][2]['G_cmd'] == 3  # own prediction neutral, not real neutral
    x = toy()
    x['z_real'][1] += .2  # threshold ~2: a's |dC|=1 is masked, c's |dC|=2 is masked
    assert evaluate(x)['candidates'][2]['G_cmd'] is None
    x = toy()
    x['ee_world'][:] = 0
    assert evaluate(x)['rankings']['all']['realized']['progress_captured'] is None
    x = toy()
    x['kind'][4] = 'onray'
    assert evaluate(x)['rankings']['offray']['realized']['selected'] is None
    # Exact threshold is retained; zero denominator and zero noise remain null.
    x = toy()
    x['z_real'][1] += .25
    x['z_real'][2] = 1.5
    assert evaluate(x)['candidates'][2]['G_cmd'] == 1.2
    x = toy()
    x['z_real'][2] = x['z_real'][0]
    assert evaluate(x)['candidates'][2]['G_cmd'] is None
    x = toy()
    x['z_pred_cmd'][3] = x['z_pred_cmd'][4]
    tied = evaluate(x)['rankings']['all']['predicted-cmd']
    assert tied['selected'] == 'b' and tied['selected_ties'] == ['b', 'c']
    for field, bad in [('ee_world', np.zeros((6, 4))), ('kind', np.array(['neutral']*6)),
                       ('z_real', np.zeros((6, 2, 1, 1, 1), np.float64)),
                       ('sim_seconds', np.full(6, np.nan)), ('images', np.zeros((6, 2, 1, 1, 3), np.uint8))]:
        x = toy()
        x[field] = bad
        try:
            validate_instance(x)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f'schema violation accepted: {field}')


def test_e000():
    import json
    from pathlib import Path
    from analyze import load_e000
    from readings import evaluate
    root = Path(__file__).resolve().parents[2]
    r = evaluate(load_e000(root), 'E000')
    evidence = json.loads((root / 'docs/preliminary-20261111/evidence.json').read_text())
    for selector, key, pairs in [('realized', 'dC_realized', 20), ('predicted-exec', 'dC_pred_executed', 20), ('predicted-cmd', 'dC_pred_commanded', 14)]:
        actual = r['rankings']['all'][selector]
        expected = evidence['selectors'][key]
        assert actual['concordance']['concordant'] == pairs == expected['pairs_ordered_like_D']
        assert actual['concordance']['total'] == 21
        assert abs(actual['regret_mm'] - expected['regret_mm']) < 1e-12
        assert abs(actual['progress_captured'] - expected['progress_captured']) < 1e-12
    cmd = r['rankings']['all']['predicted-cmd']
    assert cmd['selected'] == 'plus0.003'
    assert abs(cmd['regret_mm'] - 1.2686) < .00005
    assert abs(cmd['progress_captured'] - .2167) < .00005
    rows = {r['name']: r for r in r['candidates']}
    assert abs(rows['plus0.01']['dC_real'] - (-.029504682)) < 1e-9
    for expected in evidence['rows']:
        actual = rows[expected['action']]
        for new, old in [('dC_real', 'dC_realized'), ('dC_pred_cmd', 'dC_pred_commanded'),
                         ('dC_pred_exec', 'dC_pred_executed'), ('D_mm', 'D_mm'),
                         ('feature_response_real', 'feat_resp_realized'),
                         ('feature_response_pred_exec', 'feat_resp_pred_executed'),
                         ('cos_exec_real', 'cos_pred_executed_vs_realized')]:
            assert (actual[new] is None if expected[old] is None else abs(actual[new] - expected[old]) < 1e-12), (new, actual, expected)


def test_cli():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from tempfile import TemporaryDirectory
    from analyze import pool
    from readings import evaluate
    here = Path(__file__).resolve().parent
    (here / 'out').mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='synthetic-', dir=here / 'out') as temp:
        temp = Path(temp)
        for i in range(2):
            x = toy()
            x['ee_world'] *= i + 1
            np.savez(temp / f'instance-{i:02d}.npz', **x)
        (temp / 'meta.json').write_text(json.dumps({'synthetic': True, 'seeds': [0, 1], 'v_g': [1, 0, 0], 'controller_mode': 'fixture', 'source_commits': {}, 'weight_hashes': {}, 'timings': {}}))
        out = temp / 'scored'
        proc = subprocess.run([sys.executable, str(here / 'analyze.py'), str(temp), '--out', str(out)],
                              capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        assert proc.returncode == 0, proc.stderr
        result = json.loads((out / 'results.json').read_text())
        assert len(result['instances']) == result['pooled']['n_instances'] == 2
        p = result['pooled']['rankings']['all']['predicted-cmd']
        assert p['regret_mm'] == dict(n=2, median=3., min=2., max=4.)
        assert p['concordance']['bootstrap_ci95'] is None
        assert len((out / 'percandidate.dat').read_text().splitlines()) == 13
        assert sum(line.startswith('|---') for line in (out / 'results.md').read_text().splitlines()) == 3
        for file in ['gain_vs_amplitude.dat', 'ratio_vs_amplitude.dat', 'selectors.dat']:
            assert len((out / file).read_text().splitlines()) > 1
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        subprocess.run(proc.args, check=True, capture_output=True)
        assert before == {p.name: p.read_bytes() for p in out.iterdir()}, 'nondeterministic output'
    # A cluster bootstrap must resample instances, not candidate pairs.
    instances = [evaluate(toy(), str(i)) for i in range(4)]
    pooled = pool(instances)
    p = pooled['rankings']['all']['predicted-cmd']['concordance']
    assert p['bootstrap_ci95'] == [p['fraction'], p['fraction']]
    assert pooled['rankings']['offray']['realized']['concordance']['bootstrap_ci95'] is None
    # Three tied-cost clusters and one perfectly ordered cluster distinguish
    # whole-instance resampling from independent resampling of candidate pairs.
    flat = toy()
    flat['z_pred_cmd'][:] = 0
    mixed = [evaluate(flat, str(i)) for i in range(3)]
    perfect = toy()
    perfect['z_pred_cmd'] = np.array([4, 4, 1, 2, 3, 0], np.float32)[:, None, None, None, None].repeat(2, axis=1)
    mixed.append(evaluate(perfect, 'perfect'))
    con = pool(mixed)['rankings']['all']['predicted-cmd']['concordance']
    assert con['fraction'] == .25 and con['bootstrap_ci95'] == [0., .75]
    assert con['bootstrap_defined_draws'] == 2000


if __name__ == '__main__':
    test_cost()
    print('PASS cost')
    test_readings_contract()
    print('PASS readings: ties, own neutrals, regret, nulls, noise, schema')
    test_e000()
    print('PASS E000: realized 20/21, exec 20/21, cmd 14/21; +3mm regret/progress; all evidence rows')
    test_cli()
    print('PASS CLI: two synthetic E001 instances, all six outputs, deterministic rerun, bootstrap; fixtures deleted')
