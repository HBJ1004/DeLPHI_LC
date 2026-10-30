"""RT3c provenance, output protection and selected-family integration on CPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from repro import network_root as network
from repro import candidate_count, group_shapley, interpretation_ablations, label_shuffle_spread
from repro import amplitude_aspect_comparison, mirror_symmetry
from repro.own_period import network_controls, summarize_controls, shapley_op, nettests_results
from repro.own_period.maps import analyses, validate_damit_run


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    for key in network.VARIABLES:
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / 'artifact-root'
    records = []
    for f in range(5):
        for s in network.SEEDS:
            name = f'real-fold-{f}-seed-{s}.pt'
            path = root / 'models' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            records.append(dict(filename=name, size_bytes=path.stat().st_size, sha256=network.sha256(path)))
    dump(root / 'model-binding.json', dict(schema=network.LOCAL_SCHEMA,
         protocol_sha256=K3_PROTOCOL_SHA256, models=records))
    monkeypatch.setenv('DELPHI_NETWORK_ROOT', str(root))
    monkeypatch.setenv('DELPHI_DATA', str(tmp_path / 'fixed'))
    return root


@pytest.mark.parametrize('path,relative', [
    (label_shuffle_spread.EVALUATIONS, 'evaluations'),
    (label_shuffle_spread.OUTPUT, '../results/controls-label-shuffle/label-shuffle-spread.json'),
    (candidate_count.ENSEMBLE, 'evaluations/real-oof-ensemble.npz'),
    (candidate_count.OUTPUT, '../results/candidate-count/candidate-count.json'),
    (amplitude_aspect_comparison.OUTPUT, '../results/amplitude-aspect/amplitude-aspect.json'),
    (mirror_symmetry.LOWQ, '../results/mirror/lowq-analysis.json'),
    (group_shapley.ARTIFACTS, '.'),
    (interpretation_ablations.ARTIFACTS, '.'),
    (validate_damit_run.PUBLISHED, 'evaluations'),
])
def test_selected_paths_are_lazy(selected, path, relative):
    assert Path(path).resolve() == (selected / relative).resolve()


@pytest.mark.parametrize('folder', ['interpretation-20260923', 'interpretation-20260926',
                                   'interpretation-20260930', 'publication-revision-20260917-r2'])
def test_guard_old_interpretation_outputs(selected, folder):
    with pytest.raises(ValueError, match='published output'):
        network.guard_output(selected.parent / folder / 'result.json')


def test_evaluation_metadata_is_aligned_and_rejects_one_stale_row(selected):
    path = selected.parent / 'rows.npz'
    marker = network.evaluation_metadata(2)
    assert marker['network_binding_sha256'].shape == (2,)
    np.savez(path, object_ids=['a', 'b'], **marker)
    network.check_npz(path)
    marker['network_binding_sha256'][1] = 'old'
    np.savez(path, object_ids=['a', 'b'], **marker)
    with pytest.raises(ValueError, match='hash mismatch'):
        network.check_npz(path)


@pytest.mark.parametrize('control', ['raw', 'input-swap', 'brightness', 'amplitude', 'no-cross-term'])
def test_control_evaluator_preserves_extras_and_checks_checkpoint(selected, monkeypatch, control):
    examples = [SimpleNamespace(object_id=oid, tokenized=oid, target_axes=np.eye(3)) for oid in ('a', 'b')]
    seen = []
    def evaluate(checkpoint, rows, destination, **kwargs):
        assert Path(checkpoint) == selected / 'models/real-fold-0-seed-17.pt'
        assert kwargs['device'] == 'cpu' and kwargs['batch_size'] == 4
        metadata = kwargs['artifact_arrays']
        assert np.all(metadata['network_binding_sha256'] == network.json_metadata()['network_binding_sha256'])
        if control == 'input-swap':
            assert metadata['input_donor_ids'].tolist() == ['b', 'a']
            assert [r.tokenized for r in rows] == ['b', 'a']
        if control == 'brightness':
            assert metadata['diagnostic_control'].tolist() == [control, control]
        seen.append(True)
        return {'mean_error_deg': 1}
    monkeypatch.setattr(network_controls.run, '_evaluate_examples', evaluate)
    output = selected.parent / 'result.npz'
    assert network_controls.evaluate(control, 0, 17, examples, ['a', 'b'], output, 'cpu')['mean_error_deg'] == 1
    (selected / 'models/real-fold-0-seed-17.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        network_controls.evaluate(control, 0, 17, examples, ['a', 'b'], output, 'cpu')
    assert len(seen) == 1


def test_controls_guard_before_reading_inputs(selected, monkeypatch):
    monkeypatch.setattr('sys.argv', ['controls', '--periods', 'damit', '--out-dir',
                                    str(selected.parent / 'own-period-20261021/out')])
    with pytest.raises(ValueError, match='published output'):
        network_controls.main()


def test_controls_summary_checks_every_artifact(selected):
    folder = selected.parent / 'controls'
    folder.mkdir()
    for seed in network.SEEDS:
        for control, value in [('raw', 10), ('amplitude', 12)]:
            np.savez(folder / network_controls.output_name(0, seed, control), object_ids=['one'],
                     oracle_errors_deg=[value], **network.evaluation_metadata(1))
    result = summarize_controls.summarize(folder, [0])
    assert result['controls']['amplitude']['gap_mean_deg'] == 2
    np.savez(folder / network_controls.output_name(0, 17, 'amplitude'), object_ids=['one'], oracle_errors_deg=[12])
    with pytest.raises(ValueError, match='unbound network artifact'):
        summarize_controls.summarize(folder, [0])


@pytest.mark.parametrize('call', [lambda: summarize_controls.compare(Path('missing')),
                                  lambda: shapley_op.compute_summary(Path('missing'), True),
                                  nettests_results.published_gaps])
def test_switched_scripts_do_not_claim_old_numerical_parity(selected, call):
    with pytest.raises(ValueError, match='published'):
        call()


def test_shapley_resume_rejects_stale_file_before_loading_model(selected, monkeypatch):
    directory = selected.parent / 'shapley'
    directory.mkdir()
    dump(directory / 'period-source.json', {'period_source': 'damit', **network.json_metadata()})
    np.savez(directory / 'fold-0-seed-17.npz', network_binding_sha256='old')
    monkeypatch.setattr(shapley_op, 'fold_rows', lambda: [{'fold': 0, 'test_ids': ['one']}])
    monkeypatch.setattr(shapley_op, 'load_examples', lambda *a: pytest.fail('stale resume reached inference'))
    with pytest.raises(ValueError, match='hash mismatch'):
        shapley_op.run(SimpleNamespace(periods='damit', out_dir=directory, nside=1,
                                       device='cpu', seeds=[17], folds=[0]))


def test_shapley_summary_checks_input_binding(selected):
    path = selected.parent / 'fold-0-seed-17.npz'
    np.savez(path, network_binding_sha256='old')
    with pytest.raises(ValueError, match='hash mismatch'):
        list(shapley_op.corrected([path], {}))


def test_lowq_adapter_preserves_rows_and_records_source_hash(selected):
    source = selected.parent / 'results/lower-quality-census/scores-reference/census-analysis-excluded.json'
    rows = [{'object_id': 'damit:260', 'oracle_at_3_error_deg': 4.5, 'quality_flag': 2}]
    dump(source, {'schema': 'fresh-census', 'objects': rows, **network.json_metadata()})
    mirror_symmetry.prepare_lowq_input()
    target = selected.parent / 'results/mirror/lowq-analysis.json'
    network.check_json(target)
    result = json.loads(target.read_text())
    assert result['objects'] == rows
    assert result['schema'] == 'fresh-census'
    assert result['source_sha256'] == network.sha256(source)
    with pytest.raises(FileExistsError):
        mirror_symmetry.prepare_lowq_input()


@pytest.mark.parametrize('value', [
    {'objects': [{'object_id': 'damit:260', 'oracle_at_3_error_deg': 4}], 'network_binding_sha256': 'old'},
    {'objects': []},
    {'objects': [{'object_id': 'wrong', 'oracle_at_3_error_deg': 4}]},
    {'objects': [{'object_id': 'damit:1', 'oracle_at_3_error_deg': 4}] * 2},
])
def test_lowq_adapter_refuses_stale_or_invalid_input(selected, value):
    if value.get('network_binding_sha256') != 'old':
        value = {**value, **network.json_metadata()}
    source = selected.parent / 'results/lower-quality-census/scores-reference/census-analysis-excluded.json'
    dump(source, value)
    with pytest.raises(ValueError):
        mirror_symmetry.prepare_lowq_input()
    assert not (selected.parent / 'results/mirror/lowq-analysis.json').exists()


def test_map_analysis_outputs_carry_binding(selected):
    for name, value in [('summary.json', {'mean': 1}), ('objects.json', [{'object_id': 'one'}])]:
        path = selected.parent / name
        analyses.write_once(path, value)
        result = json.loads(path.read_text())
        if isinstance(result, list):
            result = result[0]
        assert result['network_binding_sha256'] == network.json_metadata()['network_binding_sha256']


def test_auxiliary_shuffle_uses_the_bound_audit_and_native_receipt(selected):
    checkpoint = selected / 'models/synthetic-label-shuffle-seed-17.pt'
    checkpoint.write_bytes(b'shuffle')
    audit = selected / 'checkpoint-audit.json'
    dump(audit, {'checkpoint_sha256': {checkpoint.name: network.sha256(checkpoint)}})
    binding = selected / 'model-binding.json'
    value = json.loads(binding.read_text())
    value.update(checkpoint_audit_sha256=network.sha256(audit))
    dump(binding, value)
    path = selected / 'evaluations/synthetic-label-shuffle-test-seed-17.npz'
    path.parent.mkdir()
    np.savez(path, oracle_errors_deg=[1])
    dump(Path(str(path) + '.receipt.json'), {'artifact_sha256': network.sha256(path),
                                           'checkpoint_sha256': network.sha256(checkpoint)})
    network.check_npz(path)
    checkpoint.write_bytes(b'other-run')
    with pytest.raises(ValueError, match='auxiliary checkpoint mismatch'):
        network.check_npz(path)


def test_auxiliary_shuffle_refuses_receipt_from_other_checkpoint(selected):
    checkpoint = selected / 'models/synthetic-label-shuffle-seed-17.pt'
    checkpoint.write_bytes(b'shuffle')
    audit = selected / 'checkpoint-audit.json'
    dump(audit, {'checkpoint_sha256': {checkpoint.name: network.sha256(checkpoint)}})
    binding = selected / 'model-binding.json'
    value = json.loads(binding.read_text())
    dump(binding, {**value, 'checkpoint_audit_sha256': network.sha256(audit)})
    path = selected / 'evaluations/synthetic-label-shuffle-test-seed-17.npz'
    path.parent.mkdir()
    np.savez(path, oracle_errors_deg=[1])
    dump(Path(str(path) + '.receipt.json'), {'artifact_sha256': network.sha256(path), 'checkpoint_sha256': 'old'})
    with pytest.raises(ValueError, match='auxiliary checkpoint mismatch'):
        network.check_npz(path)


def test_maps_validator_compares_selected_reference_with_unchanged_tolerances(selected, monkeypatch):
    reference = selected / 'evaluations'
    rerun = selected.parent / 'maps/evaluations'
    reference.mkdir()
    rerun.mkdir(parents=True)
    value = json.loads((selected / 'model-binding.json').read_text())
    digest = value['models'][0]['sha256']
    arrays = dict(object_ids=np.array(['one']), folds=np.array([0]),
                  checkpoint_sha256=np.array([digest]), score_grids=np.array([[1., 2., 3.]], dtype=np.float32),
                  initial_axes=np.array([np.eye(3)]), refined_axes=np.array([np.eye(3)]),
                  deployed_mode_indices=np.array([[0, 1, 2]]), oracle_errors_deg=np.array([1.]),
                  grid_oracle_errors_deg=np.array([1.]), inference_wall_seconds=np.array([.1]))
    for directory in (reference, rerun):
        np.savez(directory / 'real-oof-ensemble.npz', **arrays, **network.network_metadata())
        for f in range(5):
            for s in network.SEEDS:
                np.savez(directory / f'real-fold-{f}-seed-{s}.npz', object_ids=['one'],
                         score_grids=arrays['score_grids'], oracle_errors_deg=[1.], **network.network_metadata())
    output = selected.parent / 'map-validation.json'
    monkeypatch.setattr('sys.argv', ['validate', str(rerun.parent), str(output)])
    assert validate_damit_run.main() == 0
    result = json.loads(output.read_text())
    assert result['passed'] and result['oracle_error_max_abs_diff_deg'] == 0
    np.savez(rerun / 'real-oof-ensemble.npz', **{**arrays, 'oracle_errors_deg': np.array([1.01])},
             **network.network_metadata())
    output.unlink()
    assert validate_damit_run.main() == 1
    assert json.loads(output.read_text())['passed'] is False


def test_switched_report_uses_fresh_summaries_and_no_deferred_inputs(selected):
    for family in ('controls-input-swap', 'input-removals', 'shapley'):
        for period in ('damit', 'own'):
            result = {'groups': [{'group': 'sun', 'percentile_share': .2, 'grid_error_mean_deg': -1}]} if family == 'shapley' else {
                'summary': {'controls': {'brightness': {'gap_mean_deg': 1, 'gap_min_deg': .5, 'gap_max_deg': 2}}}}
            dump(selected.parent / 'results' / family / f'{period}-summary.json', {**result, **network.json_metadata()})
    output = selected.parent / 'RESULTS.md'
    assert nettests_results.selected_results(output) == 0
    assert 'network_binding_sha256' in output.read_text()
    assert 'Deferred families' in output.read_text()
