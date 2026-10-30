"""CPU checks for fixed-input reruns, safe bundles and network-bound resumes."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lc_pipeline.k3 import reliability_study as study
from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from repro import network_root as network
from repro.own_period import external_variants as variants, reliability as reliability
from repro.own_period import reliability_report, train_vs_test


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    for key in network.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / 'artifact-root'
    rows = []
    for fold in range(5):
        for seed in network.SEEDS:
            path = root / 'models' / f'real-fold-{fold}-seed-{seed}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
            rows.append(dict(filename=path.name, sha256=network.sha256(path), size_bytes=path.stat().st_size))
    dump(root / 'model-binding.json', dict(schema=network.LOCAL_SCHEMA,
         protocol_sha256=K3_PROTOCOL_SHA256, models=rows))
    monkeypatch.setenv('DELPHI_NETWORK_ROOT', str(root))
    monkeypatch.setenv('DELPHI_DATA', str(tmp_path / 'fixed'))
    return root


@pytest.mark.parametrize('path,relative', [(reliability.OUT, 'reliability'),
    (variants.VARIANTS_OUT, 'external-variants/ztf-variants'),
    (variants.NOERR_OUT, 'external-variants/no-uncertainty')])
def test_selected_result_paths(selected, path, relative):
    assert Path(path) == selected.parent / 'results' / relative


@pytest.mark.parametrize('path,relative', [(reliability.OUT, 'reliability'),
    (variants.VARIANTS_OUT, 'nettests/ztf-variants'), (variants.NOERR_OUT, 'nettests/no-uncertainty')])
def test_legacy_paths(selected, monkeypatch, path, relative):
    monkeypatch.delenv('DELPHI_NETWORK_ROOT')
    assert Path(path) == selected.parent / 'fixed' / relative


def test_jsonl_composite_cells_and_stale_rows(selected):
    path = selected.parent / 'rows.jsonl'
    rows = [dict(object_id='a', condition='full', repeat=i, **network.json_metadata()) for i in range(2)]
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    network.check_jsonl(path, identity_fields=('object_id', 'condition', 'repeat'))
    rows[1]['network_binding_sha256'] = 'old'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='binding mismatch'):
        network.check_jsonl(path, identity_fields=('object_id', 'condition', 'repeat'))


def make_bundle(root):
    directory = root / 'bundles/k3-oof-fold-0'
    members = []
    known = {r['filename']: r['sha256'] for r in network.binding()['models']}
    for seed in network.SEEDS:
        path = directory / f'seed-{seed}.safetensors'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(seed).encode())
        members.append(dict(seed=seed, file=path.name, sha256=network.sha256(path),
                            source_checkpoint_sha256=known[f'real-fold-0-seed-{seed}.pt']))
    dump(directory / 'bundle.json', dict(fold=0, members=members))
    dump(root / 'bundles/local-export.json', dict(**network.json_metadata(),
         file_sha256={p.relative_to(root / 'bundles').as_posix(): network.sha256(p)
                      for p in directory.iterdir()}))
    return directory


def test_safe_bundle_selected_without_opening_old_networks(selected):
    directory = make_bundle(selected)
    record = network.bundle_record(0, [])
    assert record['directory'] == str(directory)
    assert len(record['files']) == 6


@pytest.mark.parametrize('bad', ['source', 'weight', 'manifest', 'receipt'])
def test_bundle_rejects_mixed_or_mutated_networks(selected, bad):
    directory = make_bundle(selected)
    path = directory / 'bundle.json'
    if bad in ('source', 'manifest'):
        value = json.loads(path.read_text())
        value['members'][0]['source_checkpoint_sha256' if bad == 'source' else 'file'] = 'old'
        dump(path, value)
    elif bad == 'weight':
        (directory / 'seed-17.safetensors').write_bytes(b'old')
    else:
        dump(selected / 'bundles/local-export.json', dict(network_binding_sha256='old'))
    with pytest.raises((ValueError, FileNotFoundError)):
        network.bundle_record(0, [])


@pytest.mark.parametrize('call', [lambda: reliability.command_search(SimpleNamespace()),
    lambda: variants.search(SimpleNamespace()), lambda: variants.prepare_no_error(SimpleNamespace()),
    lambda: reliability_report.command_gate(SimpleNamespace())])
def test_fixed_protocol_stages_cannot_be_replaced(selected, call):
    with pytest.raises(ValueError):
        call()


def test_saved_periods_read_fixed_input(selected, monkeypatch):
    fixed = selected.parent / 'fixed/reliability/search'
    fixed.mkdir(parents=True)
    (fixed / 'periods-withheld.jsonl').write_text(json.dumps(dict(object_id='a', condition='withheld-full', repeat=0, period_hours=10)) + '\n')
    (fixed / 'periods-withheld.complete').write_text('{}')
    monkeypatch.setattr(reliability, 'OUT', selected.parent / 'new-empty-results')
    assert reliability.own_periods('withheld') == {('a', 'withheld-full', 0): 10}


def test_withheld_rerun_only_guided_four_workers_and_fixed_radius(selected, monkeypatch):
    from lc_pipeline.grid_benchmark import RADIUS_DEG
    solver = selected.parent / 'solver'
    solver.write_bytes(b'solver')
    lock = dict(objects=[dict(object_id='a', fold=0, period_hours=10)], solver=dict(executable=study.binding(solver)),
                solver_settings={}, atlases=[])
    monkeypatch.setattr(reliability, 'published_lock', lambda: lock)
    monkeypatch.setattr(reliability, 'published_schedule', lambda: dict(holdout_ids=['a']))
    monkeypatch.setattr(reliability, 'own_periods', lambda scope: {('a', 'withheld-' + s, 0): 10 for s in reliability.WITHHELD_SAMPLINGS})
    for sampling in reliability.WITHHELD_SAMPLINGS:
        study.seal(Path(reliability.withheld_root('own')) / 'fits/a' / sampling / 'k3/prediction.json', network.json_metadata())
    seen = []
    def fit(root, row, sampling, arm, selected_lock, predictor, budget, *, workers):
        seen.append((arm, workers))
        assert selected_lock['network_binding_sha256'] == network.json_metadata()['network_binding_sha256']
        return dict(status='ok')
    monkeypatch.setattr(study, 'fit_cell', fit)
    args = SimpleNamespace(mode='own', ids=None, arms=None, samplings=None, max_seconds=100)
    assert reliability.command_withheld_fits(args) == 0
    assert seen == [('k3', 4), ('k3', 4)]
    assert RADIUS_DEG == 15.772555008039914
    with pytest.raises(ValueError, match='reuse classical20'):
        reliability.command_withheld_fits(SimpleNamespace(**{**vars(args), 'arms': ['classical20']}))


def test_stale_sealed_prediction_rejected_before_resume(selected):
    path = selected.parent / 'predictions/a/full/repeat-0.json'
    study.seal(path, dict(object_id='a', condition='full', repeat=0, fold=0, seed=study.SEEDS[0],
        source_sha256='x', model_count=5, model_id='k3', network_binding_sha256='old'))
    with pytest.raises(ValueError, match='identity mismatch'):
        study.prediction_cell(selected.parent, dict(object_id='a', fold=0, lightcurve=dict(sha256='x')),
                              dict(condition='full'), 0, None, network_metadata=network.json_metadata())


def test_stale_guided_fit_rejected_before_solver(selected):
    path = selected.parent / 'fits/a/full/k3/result.json'
    study.seal(path, dict(network_binding_sha256='old'))
    with pytest.raises(ValueError, match='network binding mismatch'):
        study.fit_cell(selected.parent, dict(object_id='a'), 'full', 'k3', network.json_metadata(), None, None, workers=4)


def test_role_report_rejects_old_parity_claim(selected):
    with pytest.raises(ValueError, match='historical numerical parity'):
        train_vs_test.report(SimpleNamespace(output=selected.parent / 'new.json', compare_published=True))


def test_period_prepare_keeps_exact_conditions_and_schedule(selected, monkeypatch):
    old = selected.parent / 'published'
    lightcurve = selected.parent / 'lc.txt'
    lightcurve.write_text('fixed photometry')
    catalog = selected.parent / 'catalog.jsonl'
    catalog.write_text('fixed catalog')
    conditions = study.conditions()
    schedule = dict(repeats=3, holdout_ids=['a'], unchanged=['schedule', 17])
    study.seal(old / 'study.json', dict(objects=[dict(object_id='a', lightcurve=study.binding(lightcurve))],
               conditions=conditions, bundles=[], catalog=study.binding(catalog)))
    study.seal(old / 'schedule.json', schedule)
    monkeypatch.setattr(network, 'bundle_record', lambda fold, bundles: dict(fold=fold, directory=str(selected / 'bundles' / str(fold))))
    root = selected.parent / 'period-new'
    study.prepare_period_rerun(root=root, published_root=old)
    lock = study.unseal(root / 'study.json')
    assert lock['conditions'] == [c for c in conditions if c['family'] == 'period' or c['condition'] == 'full']
    assert study.unseal(root / 'schedule.json') == schedule == lock['fixed_schedule']
    assert lock['publication_promotion_claimed'] is False
    assert study._period_rerun_lock(root) == lock
    (root / 'schedule.json').unlink()
    study.seal(root / 'schedule.json', {'repeats': 1})
    with pytest.raises(ValueError, match='schedule changed'):
        study._period_rerun_lock(root)


def test_period_score_requires_completed_bound_inference(selected, monkeypatch):
    monkeypatch.setattr(study, '_period_rerun_lock', lambda root: {})
    root = selected.parent / 'period-new'
    study.seal(root / 'inference-complete.json', dict(network_binding_sha256='old'))
    with pytest.raises(ValueError, match='binding hash mismatch'):
        study.score_period_rerun(root=root)


def test_local_bundle_export_preserves_tensors_and_records_skipped_gates(selected, monkeypatch):
    import torch
    from safetensors.torch import load_file
    from repro import export_k3_bundles as exporter
    from repro import prepare_k3_followup_inputs as inputs
    repository = Path(__file__).resolve().parents[1]
    splits = repository / 'repro/data/damit-20250610T000301Z/publication-splits-v2.3.json'
    records, hashes, names = [], [], []
    state = {'weight': torch.tensor([[1., -2.], [3., 4.]])}
    for fold in range(5):
        for seed in network.SEEDS:
            path = selected / 'models' / f'real-fold-{fold}-seed-{seed}.pt'
            checkpoint = dict(stage='real-oof', config=dict(seed=seed),
                              run_provenance=dict(implementation_commit='training', protocol_sha256=K3_PROTOCOL_SHA256),
                              model_config={}, model_state_dict=state)
            torch.save(checkpoint, path)
            records.append(dict(filename=path.name, sha256=network.sha256(path), size_bytes=path.stat().st_size,
                                run_provenance=checkpoint['run_provenance']))
            names.append(path.name)
            hashes.append(network.sha256(path))
    evaluation = selected / 'evaluations'
    evaluation.mkdir()
    seed_hashes = []
    for fold in range(5):
        for seed in network.SEEDS:
            path = evaluation / f'real-fold-{fold}-seed-{seed}-calibration.npz'
            np.savez(path, seed=seed)
            seed_hashes.append(network.sha256(path))
            dump(Path(str(path) + '.receipt.json'), dict(artifact_sha256=network.sha256(path),
                 checkpoint_sha256=hashes[names.index(path.name.replace('-calibration.npz', '.pt'))]))
    ensemble = evaluation / 'real-oof-ensemble.npz'
    np.savez(ensemble, checkpoint_names=names, checkpoint_sha256=hashes)
    cal = evaluation / 'real-calibration-ensemble.npz'
    np.savez(cal, input_calibration_sha256=seed_hashes, checkpoint_names=names, checkpoint_sha256=hashes)
    calibrated = evaluation / 'real-oof-ensemble-calibrated.npz'
    np.savez(calibrated, calibration_ensemble_sha256=network.sha256(cal), oof_ensemble_sha256=network.sha256(ensemble),
             fold_cone90_deg=np.arange(5) + 30, fold_cone95_deg=np.full(5, 90))
    for path in (cal, calibrated):
        dump(Path(str(path) + '.receipt.json'), dict(artifact_sha256=network.sha256(path)))
    dump(selected / 'model-binding.json', dict(schema=network.LOCAL_SCHEMA, protocol_sha256=K3_PROTOCOL_SHA256,
        models=records, artifact_sha256={'evaluations/real-oof-ensemble.npz': network.sha256(ensemble)}))
    monkeypatch.setattr(exporter, 'K3EnsemblePredictor', lambda directory: None)
    monkeypatch.setattr(exporter.subprocess, 'check_output', lambda *a, **kw: 'analysis-head')
    output = selected / 'bundles'
    result = exporter.export(selected, splits, output, repository, local_rerun=True)
    assert result['tensor_parity'] == 'exact'
    assert result['publication_promotion_claimed'] is False
    assert len(result['skipped_publication_gates']) == 3
    for fold in range(5):
        record = network.bundle_record(fold, [])
        for seed in network.SEEDS:
            assert torch.equal(load_file(str(Path(record['directory']) / f'seed-{seed}.safetensors'))['weight'], state['weight'])
    # The receipt is verified independently rather than using a publication summary.
    assert inputs.local_evaluation_receipt(selected, calibrated)['artifact_sha256'] == network.sha256(calibrated)


def test_period_rerun_resume_and_score_use_fixed_cells(selected, monkeypatch):
    from lc_pipeline.k3 import bundle
    from lc_pipeline.k3.reliability_sampling import CurveObservation, LightCurve, NativeSession, write_lc
    from lc_pipeline.v2 import data
    source = selected.parent / 'period-source'
    rows = []
    for fold in range(5):
        oid = f'object-{fold}'
        points = tuple(CurveObservation(2450000. + j / 50, 1. + j / 100,
                       (1., 0., 0.), (0., 1., 0.)) for j in range(8))
        curve = LightCurve((NativeSession('s', points, calibrated=0, source_index=0),), oid, 10.)
        path = selected.parent / f'{oid}.lc'
        write_lc(curve, path)
        rows.append(dict(object_id=oid, fold=fold, period_hours=10., lightcurve=study.binding(path)))
    catalog = selected.parent / 'catalog'
    catalog.write_text('fixed catalog')
    study.seal(source / 'study.json', dict(objects=rows, conditions=study.conditions(), bundles=[], catalog=study.binding(catalog)))
    study.seal(source / 'schedule.json', dict(repeats=3))
    monkeypatch.setattr(network, 'bundle_record', lambda fold, bundles: dict(directory=str(selected / str(fold))))
    seen = []
    def predict(epochs, *, known_period, object_id):
        seen.append((object_id, known_period.hours))
        return dict(axes=[dict(axis_xyz=axis) for axis in np.eye(3).tolist()])
    monkeypatch.setattr(bundle, 'K3EnsemblePredictor', lambda directory, device: SimpleNamespace(predict=predict))
    monkeypatch.setattr(data, 'load_catalog', lambda path: {r['object_id']: SimpleNamespace(solution_vectors=[[1., 0., 0.]]) for r in rows})
    root = selected.parent / 'period-new'
    study.prepare_period_rerun(root=root, published_root=source)
    assert study.run_period_rerun(root=root, device='cpu', max_seconds=0)['remaining'] == 45
    assert study.run_period_rerun(root=root, device='cpu', max_seconds=100)['remaining'] == 0
    assert len(seen) == 45
    assert {hours for _, hours in seen} == {10., *(10. * factor for factor in study.PERIOD_MULTIPLIERS)}
    # A resumed completed run reads and validates cached cells, making no prediction.
    assert study.run_period_rerun(root=root, device='cpu', max_seconds=100)['remaining'] == 0
    assert len(seen) == 45
    report = study.score_period_rerun(root=root)
    assert report['sampling']['models']['k3']['full']['mean_error_deg'] == 0
    assert len(study.unseal(root / 'scored-rows.json')) == 45
