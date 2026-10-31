"""CPU-only protocol and provenance checks for RT3d1 adapters."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from repro import network_root as network
from repro.direct_workflow_benchmark import rerun
from repro.own_period import temporal, refine_chain_post as post, v6_fixed_period as v6, v7_grid as v7
from lc_pipeline import grid_benchmark as grid
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    for key in network.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / 'artifact-root'
    records = []
    for fold in range(5):
        for seed in network.SEEDS:
            path = root / 'models' / f'real-fold-{fold}-seed-{seed}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
            records.append(dict(filename=path.name, sha256=network.sha256(path), size_bytes=path.stat().st_size))
    dump(root / 'model-binding.json', dict(schema=network.LOCAL_SCHEMA,
         protocol_sha256=K3_PROTOCOL_SHA256, models=records))
    monkeypatch.setenv('DELPHI_NETWORK_ROOT', str(root))
    monkeypatch.setenv('DELPHI_DATA', str(tmp_path / 'fixed'))
    return root


@pytest.mark.parametrize('value,relative', [(temporal.OUTPUT, 'temporal'),
    (post.OUTPUT, 'refine-chain'), (post.BENCHMARK, 'delphi-inversions/own'),
    (post.PUBLISHED, 'delphi-inversions/damit')])
def test_family_paths_use_central_switch(selected, value, relative):
    assert Path(value) == selected.parent / 'results' / relative


@pytest.mark.parametrize('folder', ['benchmark', 'direct-workflow-benchmark', 'pole-grid-benchmark'])
def test_old_outputs_refused(selected, folder):
    path = selected.parent / ('own-period-20261021' if folder == 'benchmark' else folder) / 'score.json'
    with pytest.raises(ValueError, match='published output'):
        rerun.save(path, {})


def test_temporal_legacy_manifest_preserves_nonstandard_data_layout(selected, monkeypatch):
    monkeypatch.delenv('DELPHI_NETWORK_ROOT')
    assert Path(temporal.MODEL_MANIFEST) == selected.parent / 'external-models-final.json'
    assert repr(post.BENCHMARK).startswith('DataPath(')


@pytest.mark.parametrize('call', [lambda: temporal.command_search(SimpleNamespace()),
    lambda: __import__('repro.own_period.refine_chain', fromlist=['command_refine']).command_refine(SimpleNamespace()),
    lambda: __import__('repro.own_period.full_grid', fromlist=['main']).main()])
def test_independent_searches_and_full_grid_reused(selected, call):
    with pytest.raises(ValueError, match='reuse'):
        call()


def make_source(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    lc = source / 'lc.txt'
    lc.parent.mkdir()
    lc.write_text('photometry')
    exe = source / 'convexinv'
    exe.write_bytes(b'solver')
    catalog = source / 'catalog.jsonl'
    catalog.write_text('reference')
    code = source / 'native.py'
    code.write_text('faithful')
    monkeypatch.setattr(rerun, 'SOURCES', source)
    monkeypatch.setattr(rerun, 'HASHES', {'native.py': network.sha256(code)})
    monkeypatch.setattr(network, 'bundle_record', lambda fold, old: {'fold': fold, 'directory': f'chosen-{fold}'})
    monkeypatch.setattr(grid, '_environment', lambda device: {
        'device': device, 'cpu_models': ['new CPU'], 'gpu': ['new GPU']})
    lock = dict(objects=[dict(object_id='a', fold=0, period_hours=10.123456, lightcurve=rerun.bound(lc))],
        settings=dict(workers=6, repeats=3, seed=20260923, bootstrap_resamples=10000),
        solver_settings={'timeout_seconds_per_start': 300, 'iteration_cap': 1000},
        solver={'executable': rerun.bound(exe)}, bundles=[], selected_classical_starts=12,
        classical_starts=[[1, 0, 0]] * 12, environment={'cpu_models': ['old CPU'], 'gpu': ['old GPU']})
    dump(source / 'evaluation-lock.json', lock)
    dump(source / 'scoring-lock.json', dict(evaluation_lock=rerun.bound(source / 'evaluation-lock.json'),
                                           reference_catalog=rerun.bound(catalog)))
    return source, lock


@pytest.mark.parametrize('timing', ['reuse', 'retime'])
@pytest.mark.parametrize('original_workers', [2, 6, 9])
def test_prepare_freezes_protocol_and_old_worker_count(selected, tmp_path, monkeypatch, timing, original_workers):
    source, old = make_source(tmp_path, monkeypatch)
    old['settings']['workers'] = original_workers
    dump(source / 'evaluation-lock.json', old)
    output = tmp_path / 'new'
    lock = rerun.prepare(source, output, 'cpu', timing)
    for field in ('objects', 'settings', 'solver_settings', 'classical_starts'):
        assert lock[field] == old[field]
    workers = original_workers
    assert lock['rerun']['solver_workers'] == workers
    assert rerun.read(output / 'scoring-lock.json')['solver_workers'] == workers
    assert lock['rerun']['classical_timing'] == timing
    assert rerun.validate(output / 'evaluation-lock.json') == lock
    assert rerun.prepare(source, output, 'cpu', timing) == lock


@pytest.mark.parametrize('field', ['objects', 'settings', 'solver_settings', 'classical_starts'])
def test_lock_rejects_protocol_drift(selected, tmp_path, monkeypatch, field):
    source, _ = make_source(tmp_path, monkeypatch)
    output = tmp_path / 'new'
    lock = rerun.prepare(source, output, 'cpu')
    lock[field] = []
    dump(output / 'evaluation-lock.json', lock)
    with pytest.raises(ValueError, match='fixed benchmark'):
        rerun.validate(output / 'evaluation-lock.json')


def test_retained_original_order_filters_without_reranking():
    lock = dict(objects=[dict(object_id=str(i), fold=i % 5) for i in range(10)])
    for mode, expected in [('cold', 60), ('loaded', 60), ('exploratory', 30)]:
        jobs = rerun.jobs(lock, mode)
        assert len(jobs) == expected
        assert jobs == rerun.jobs(lock, mode)
        if mode != 'cold':
            assert [j['fold'] for j in jobs] == sorted(j['fold'] for j in jobs)


@pytest.mark.parametrize('field', ['initial_axis', 'selected_fit', 'files', 'iterations', 'failure_reason'])
def test_retiming_rejects_any_fit_or_product_change(field):
    old = dict(start_index=0, initial_axis=[1, 0, 0], selected_fit={'rms': .1},
               files={'stdout': 'digest'}, iterations=20, failure_reason=None)
    rerun.equal_start({**old, 'solver_wall_seconds': 100}, {**old, 'solver_wall_seconds': 1})
    new = deepcopy(old)
    new[field] = 'changed'
    with pytest.raises(ValueError, match='retimed classical'):
        rerun.equal_start(new, old)


def test_saved_product_hash_is_checked(tmp_path):
    product = tmp_path / 'stdout'
    product.write_text('old')
    dump(tmp_path / 'record.json', dict(files={'stdout': network.sha256(product)}))
    rerun.audit_start(tmp_path / 'record.json')
    product.write_text('changed')
    with pytest.raises(ValueError, match='product changed'):
        rerun.audit_start(tmp_path / 'record.json')


def test_v6_uses_explicit_old_arm_order(selected):
    for oid, arms in [('a', ['candidate', 'baseline']), ('b', ['baseline', 'candidate'])]:
        dump(selected.parent / 'fixed/v67/v6-own-period/runs' / oid / 'object-done.json', {'arm_order': arms})
    assert v6.arm_order(['b', 'a']) == {'a': ('candidate', 'baseline'), 'b': ('baseline', 'candidate')}


def test_v6_rejects_mixed_network_axes(selected, tmp_path):
    path = tmp_path / 'axes.npz'
    np.savez(path, object_ids=np.array(['a']), refined_axes=np.ones((1, 3, 3)), network_binding_sha256='old')
    with pytest.raises(ValueError, match='binding hash mismatch'):
        v6.load_axes(path, ['a'])


@pytest.mark.parametrize('timing', ['reuse', 'retime'])
@pytest.mark.parametrize('original_workers', [6, 9])
def test_v7_reuses_exact_old_periods_and_worker_choice(selected, tmp_path, monkeypatch, timing, original_workers):
    path = selected.parent / 'fixed/v67/v7-own-period-lock.json'
    original = dict(objects=[dict(object_id='a', period_hours=10.123456)], settings={'workers': original_workers},
                    bundles=[], device='cpu', solver={}, inputs={})
    dump(path, original)
    period = tmp_path / 'periods.json'
    dump(period, {'periods_hours': {'a': 10.123456}})
    monkeypatch.setattr(network, 'bundle_record', lambda f, old: {'fold': f})
    monkeypatch.setattr(grid, '_environment', lambda device: {})
    args = SimpleNamespace(periods=str(period), output=tmp_path / 'lock.json', device='cpu', classical_timing=timing)
    assert v7.rerun_lock(args) == 0
    actual = rerun.read(args.output)
    assert actual['objects'] == original['objects']
    assert actual['settings']['workers'] == original_workers
    assert v7.validate_rerun_lock(args.output) == actual
    actual['settings']['workers'] += 1
    dump(args.output, actual)
    with pytest.raises(ValueError, match='settings changed'):
        v7.validate_rerun_lock(args.output)
    dump(period, {'periods_hours': {'a': 11}})
    with pytest.raises(ValueError, match='periods changed'):
        v7.rerun_lock(args)


@pytest.mark.parametrize('timing', ['reuse', 'retime'])
def test_direct_worker_drift_and_explicit_mismatch_rejected(selected, tmp_path, monkeypatch, timing):
    source, _ = make_source(tmp_path, monkeypatch)
    expected = 6
    with pytest.raises(ValueError, match='workers differ'):
        rerun.prepare(source, tmp_path / 'bad', 'cpu', timing, workers=expected + 1)
    output = tmp_path / 'new'
    value = rerun.prepare(source, output, 'cpu', timing, workers=expected)
    value['rerun']['solver_workers'] += 1
    dump(output / 'evaluation-lock.json', value)
    with pytest.raises(ValueError, match='solver workers changed'):
        rerun.validate(output / 'evaluation-lock.json')


@pytest.mark.parametrize('timing', ['reuse', 'retime'])
def test_v7_explicit_cpu_validation_keeps_four_workers(selected, tmp_path, monkeypatch, timing):
    source = tmp_path / 'original.json'
    dump(source, dict(objects=[], settings={'workers': 6}, bundles=[], device='cuda', solver={}, inputs={}))
    monkeypatch.setattr(v7, 'PUBLISHED_LOCK', source)
    monkeypatch.setattr(network, 'bundle_record', lambda f, old: {'fold': f})
    monkeypatch.setattr(grid, '_environment', lambda device: {})
    args = SimpleNamespace(periods='published', output=tmp_path / 'cpu.json', device='cpu', workers=4,
                           classical_timing=timing)
    v7.rerun_lock(args)
    actual = v7.validate_rerun_lock(args.output)
    assert actual['settings']['workers'] == 4
    assert actual['rerun']['untimed_validation'] is True
    args.device, args.workers = 'cuda', 3
    with pytest.raises(ValueError, match='timing protocol'):
        v7.rerun_lock(args)


@pytest.mark.parametrize('timing', ['reuse', 'retime'])
@pytest.mark.parametrize('original_workers', [6, 12])
def test_refine_lock_and_fit_plan_use_original_plan_workers(selected, tmp_path, monkeypatch, timing, original_workers):
    original_plan = selected.parent / 'fixed/refine-chain/fits/fit-plan.json'
    dump(original_plan, {'workers': original_workers})
    refined_path = selected.parent / 'fixed/refine-chain/refined-periods.json'
    dump(refined_path, {})
    benchmark = tmp_path / 'benchmark'
    dump(benchmark / 'evaluation-lock.json', {})
    protocol = tmp_path / 'PROTOCOL.md'
    protocol.write_text('fixed protocol')
    parent = dict(rerun={'classical_timing': timing, 'solver_workers': 6})
    refined = {'periods_hours': {'a': 10}}
    objects = [{'object_id': 'a'}]
    monkeypatch.setattr(post, 'refined_inputs', lambda args: (parent, refined, objects))
    cpus = tuple(sorted(os.sched_getaffinity(0)))
    args = SimpleNamespace(output_dir=tmp_path / 'output', benchmark_dir=benchmark, protocol=protocol,
                           cpus=cpus, workers=None, classical_timing=timing)
    lock = post.lock_outputs(args)[args.output_dir / 'evaluation-lock-refined.json']
    expected = original_workers
    assert lock['rerun']['solver_workers'] == expected
    assert post.verify(lock['rerun']['source_worker_plan']) == original_plan
    assert parent['rerun']['solver_workers'] == 6
    monkeypatch.setattr(post, '_validate_runtime', lambda value: None)
    monkeypatch.setattr(post, 'checked_lock', lambda args: lock)
    monkeypatch.setattr(post, 'load_predictions', lambda *args: {})
    monkeypatch.setattr(post, 'fit_tasks', lambda *args: [])
    monkeypatch.setattr(post, 'ladder_indexes', lambda *args: [])
    monkeypatch.setattr(post, '_fit_completion', lambda *args: {})
    assert post.command_fits(args) == 0
    plan = post.read(args.output_dir / 'fits/fit-plan.json')
    assert plan['workers'] == (expected if timing == 'reuse' else 4)
    if timing == 'reuse':
        args.workers = 4
        with pytest.raises(ValueError, match='original refine plan'):
            post.command_fits(args)
        args.workers, args.cpus = None, cpus[:4]
        with pytest.raises(ValueError, match='number of CPUs'):
            post.command_fits(args)


@pytest.mark.parametrize('timing', ['reuse', 'retime'])
def test_v6_records_original_serial_worker_setting(selected, tmp_path, monkeypatch, timing):
    monkeypatch.setattr(v6, 'digest', lambda path: v6.EXECUTABLE_SHA256)
    monkeypatch.setattr(v6, 'object_order', lambda: [])
    monkeypatch.setattr(v6, 'damit_periods', lambda: {})
    monkeypatch.setattr(v6, 'load_axes', lambda *args: {})
    monkeypatch.setattr(v6, 'lightcurve_paths', lambda: {})
    monkeypatch.setattr(network, 'check_npz', lambda *args: None)
    args = SimpleNamespace(output=tmp_path / 'v6', periods='damit', axes=tmp_path / 'axes.npz',
                           ids=None, max_seconds=0, workers=1, classical_timing=timing)
    assert v6.run(args) == 0
    assert rerun.read(args.output / 'run-header.json')['solver_workers'] == 1
    args.workers = 6
    with pytest.raises(ValueError, match='serial solver setting'):
        v6.run(args)


def test_refine_classical_tasks_reference_only_saved_starts(selected, tmp_path, monkeypatch):
    directory = tmp_path / 'refine'
    dump(directory / 'evaluation-lock-refined.json', {})
    dump(directory / 'predictions-refined/predictions.json', {})
    for arm in ('standard6', 'classical20'):
        for i in range(6 if arm == 'standard6' else 146):
            dump(selected.parent / 'fixed/refine-chain/fits' / arm / 'a' / f'start-{i:03d}/timing.json', {})
    lock = {'objects': [{'object_id': 'a', 'period_hours': 10, 'fold': 0}]}
    predictions = {'a': {'refined_axes': np.eye(3).tolist()}}
    args = SimpleNamespace(output_dir=directory, classical_timing='reuse')
    tasks = post.fit_tasks(args, lock, predictions)
    assert len(tasks) == 158
    classical = [t for t in tasks if t['arm'] != 'delphi']
    assert all('classical_source' in t for t in classical)
    assert all(Path(t['directory']).is_relative_to(selected.parent / 'fixed') for t in classical)
    args.classical_timing = 'retime'
    tasks = post.fit_tasks(args, lock, predictions)
    assert all(Path(t['directory']).is_relative_to(directory) for t in tasks)


def test_central_bundle_is_checked_before_predictor_load(selected, monkeypatch):
    called = []
    def fail(fold, bundles):
        called.append(fold)
        raise ValueError('selected bundle mismatch')
    monkeypatch.setattr(network, 'bundle_record', fail)
    with pytest.raises(ValueError, match='selected bundle'):
        grid._predictor({'device': 'cpu', 'bundles': []}, 2)
    assert called == [2]


def test_execution_duplicate_or_missing_cases_refused(selected, tmp_path, monkeypatch):
    source, _ = make_source(tmp_path, monkeypatch)
    output = tmp_path / 'new'
    rerun.prepare(source, output, 'cpu')
    execution = output / 'evaluation/execution.json'
    dump(execution, dict(lock_sha256=network.sha256(output / 'evaluation-lock.json'), rows=[], mode='cold',
                         **network.json_metadata()))
    with pytest.raises(ValueError, match='incomplete'):
        rerun.verify_execution(output / 'evaluation-lock.json', execution)


def test_comparator_membership_reports_failed_cells_before_aggregation():
    from repro.own_period.derived_comparators import membership_changes
    old = [dict(object_id='a', condition='full', repeat=0, status='ok'),
           dict(object_id='a', condition='full', repeat=1, status='ok')]
    new = [old[0], {**old[1], 'status': 'failed'}]
    ztf = [dict(object_id='z', own_error_deg=10.)]
    assert not membership_changes(old, old, ztf, ztf)['membership_changed']
    report = membership_changes(old, new, ztf, [{**ztf[0], 'own_error_deg': None}])
    assert report['membership_changed']
    assert report['reliability_changes']['full']['lost'] == [('a', 1)]
    assert report['ztf_changes']['lost'] == ['z']


@pytest.mark.parametrize('key', ['final_lambda_deg', 'output_lightcurve_sha256', 'iterations'])
def test_v6_retiming_ignores_only_time_and_provenance(key):
    old = {'identity': {'object_id': 'a'}, 'result': {'final_lambda_deg': 1.,
        'output_lightcurve_sha256': 'same', 'iterations': 50, 'wall_time_seconds': 1}}
    new = deepcopy(old)
    new['result']['wall_time_seconds'] = 2
    rerun.equal_downstream(new, old)
    new['result'][key] = 'different'
    with pytest.raises(ValueError, match='downstream fit differs'):
        rerun.equal_downstream(new, old)


@pytest.mark.parametrize('timing', ['reuse', 'retime'])
@pytest.mark.parametrize('workers', [1, 4, 6, 12])
def test_timed_workers_keeps_original_concurrency(timing, workers):
    assert rerun.timed_workers(workers, timing) == workers


@pytest.mark.parametrize('workers', [0, -1, True, 6.0, '6', None])
def test_timed_workers_rejects_invalid_original_count(workers):
    with pytest.raises(ValueError, match='positive integer'):
        rerun.timed_workers(workers, 'retime')


def test_timed_workers_rejects_unknown_mode():
    with pytest.raises(ValueError, match='unknown classical timing'):
        rerun.timed_workers(6, 'both')


def write_execution(output, lock, mode):
    folder = {'cold': 'evaluation', 'loaded': 'evaluation-loaded',
              'exploratory': 'exploratory-loaded-ladder'}[mode]
    directory = output / folder
    rows = []
    for job in rerun.jobs(lock, mode):
        path = directory / 'cases' / f"{job['object_id']}-r{job['repeat_index']}-{job['arm']}" / 'timed-result.json'
        dump(path, {**job, 'start_records': [], 'retiming_cpu_seconds': 2,
                    **network.json_metadata()})
        rows.append(dict(path=str(path.relative_to(directory)), sha256=network.sha256(path)))
    dump(directory / 'execution.json', dict(mode=mode, rows=rows,
         lock_sha256=network.sha256(output / 'evaluation-lock.json'), **network.json_metadata()))
    return directory / 'execution.json'


@pytest.fixture
def exploratory_reuse(selected, tmp_path, monkeypatch):
    source, _ = make_source(tmp_path, monkeypatch)
    reuse, target = tmp_path / 'reuse', tmp_path / 'retime'
    old = rerun.prepare(source, reuse, 'cpu', 'reuse')
    lock = rerun.prepare(source, target, 'cpu', 'retime')
    execution = write_execution(reuse, old, 'exploratory')
    saved = reuse / 'classical-ladder-exploratory.json'
    dump(saved, dict(object_rows=[], **network.json_metadata()))
    dump(reuse / 'exploratory-loaded-ladder/score.json', dict(
        execution_sha256=network.sha256(execution), saved_fit_sha256=network.sha256(saved),
        arm_summaries=[dict(arm='classical18', mean_wall_seconds=18)],
        solver_workers=6, **network.json_metadata()))
    return source, reuse, target, lock


def test_exploratory_reuse_audits_original_lock_and_records_mixed_hardware(exploratory_reuse):
    _, reuse, _, lock = exploratory_reuse
    score, metadata = rerun.reused_exploratory(reuse, lock)
    assert score == rerun.read(reuse / 'exploratory-loaded-ladder/score.json')
    assert metadata['status'] == 'reused_not_retimed'
    assert metadata['classical_environment']['cpu_models'] == ['old CPU']
    assert metadata['delphi_environment']['gpu'] == ['new GPU']
    assert 'old machine' in metadata['note']


@pytest.mark.parametrize('change', ['case', 'score_binding', 'saved_fits', 'network', 'period', 'mode'])
def test_exploratory_reuse_rejects_changed_or_mixed_evidence(exploratory_reuse, change):
    _, reuse, _, lock = exploratory_reuse
    if change == 'case':
        path = next((reuse / 'exploratory-loaded-ladder/cases').rglob('timed-result.json'))
        dump(path, {})
    elif change in ('score_binding', 'network'):
        path = reuse / 'exploratory-loaded-ladder/score.json'
        doc = rerun.read(path)
        doc['execution_sha256' if change == 'score_binding' else 'network_binding_sha256'] = 'wrong'
        dump(path, doc)
    elif change == 'saved_fits':
        path = reuse / 'classical-ladder-exploratory.json'
        dump(path, {**rerun.read(path), 'changed': True})
    elif change == 'period':
        lock['objects'][0]['period_hours'] += 1
    else:
        lock['rerun']['classical_timing'] = 'reuse'
    with pytest.raises(ValueError):
        rerun.reused_exploratory(reuse, lock)


def test_statistics_reuses_ladder_without_retiming_or_changing_source(exploratory_reuse, monkeypatch):
    source, reuse, target, lock = exploratory_reuse
    for mode in ('cold', 'loaded'):
        write_execution(target, lock, mode)
    archive = source / 'full-grid'
    for arm, count in [('standard6', 6), ('classical20', 146)]:
        for index in range(count):
            dump(archive / arm / f'start-{index:03d}/record.json', {'files': {}})
    monkeypatch.setattr(rerun.prior, '_archived_case', lambda root, oid, repeat, arm: root / arm)
    monkeypatch.setattr(rerun, 'modules', lambda: (SimpleNamespace(
        score=lambda lock, scoring, execution, output: dump(output, {'unchanged_statistic': 7})), None))
    called = []
    def fake_producer(name, **overrides):
        called.append(name)
        assert name != 'score_exploratory_timing.py'
        return SimpleNamespace(main=lambda: dump(overrides['OUTPUT'], {'unchanged_statistic': 9}))
    monkeypatch.setattr(rerun, 'producer', fake_producer)
    before = {p: p.read_bytes() for p in reuse.rglob('*') if p.is_file()}
    rerun.analyses(target, reuse)
    assert before == {p: p.read_bytes() for p in reuse.rglob('*') if p.is_file()}
    assert not (target / 'exploratory-loaded-ladder/execution.json').exists()
    for filename in ('score.json', 'score-loaded.json', 'timing-cost.json'):
        doc = rerun.read(target / filename)
        assert doc['environment']['cpu_models'] == ['new CPU']
        assert doc['environment']['gpu'] == ['new GPU']
        assert doc['exploratory_timing']['classical_environment']['cpu_models'] == ['old CPU']
    assert rerun.read(target / 'timing-cost.json')['classical_retiming_cpu_seconds'] == 12
    copied = rerun.read(target / 'exploratory-loaded-ladder/score.json')
    assert copied['arm_summaries'] == rerun.read(reuse / 'exploratory-loaded-ladder/score.json')['arm_summaries']
    assert rerun.read(target / 'score.json')['unchanged_statistic'] == 7
    assert len(called) == 4
