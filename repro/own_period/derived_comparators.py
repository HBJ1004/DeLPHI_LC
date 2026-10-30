"""Faithful replay of the published successful-subset comparators.

Mirrors data/own-period-20261021/failed-excluded-derived/derive.py
SHA256: e0c07f603a3ec9bb58de2cef98c469e1a50b2f87d2cee67ffb5e8cf440194c8a
Only path selection, provenance, membership auditing and copying the unchanged
ALCDEF row differ. The original seeds, random draws and estimands are retained.
"""
import argparse
import shutil
from repro import network_root as network
from repro.own_period import DataPath
SOURCE_SHA256 = "e0c07f603a3ec9bb58de2cef98c469e1a50b2f87d2cee67ffb5e8cf440194c8a"
"""Audit paired successful subsets using frozen rows only. No search or inference."""
import ast, json
from collections import defaultdict
from pathlib import Path
import numpy as np
F=DataPath('../..')
D=F/'data/own-period-20261021'
O=None
def read(p): return json.loads(p.read_text())
def save(name,x):
 network.guard_output(O/name)
 if network.switched():
  x.update(network.json_metadata())
  paths=[network.NetworkPath('results','reliability','score/scored-rows-own.json'),
         network.NetworkPath('results','reliability','score/reliability-own-period.json'),
         network.NetworkPath('results','ztf','own-period-scores.json')]
  x['selected_input_sha256']={str(p):network.sha256(p) for p in paths}
  if 'sources' in x:
   x['published_source_descriptions']=x.pop('sources')
   x['sources']=[str(p) for p in paths]
 (O/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load_functions(p,names,namespace):
 s=p.read_text();t=ast.parse(s)
 for node in t.body:
  if isinstance(node,ast.FunctionDef) and node.name in names: exec(compile(ast.get_source_segment(s,node),str(p),'exec'),namespace)

def membership_changes(old_rows, rows, old_ztf, ztf):
 changes={}
 for condition in sorted({r['condition'] for r in old_rows+rows}):
  old={(r['object_id'],r['repeat']) for r in old_rows if r['condition']==condition and r['status']=='ok'}
  new={(r['object_id'],r['repeat']) for r in rows if r['condition']==condition and r['status']=='ok'}
  if old!=new:changes[condition]={'lost':sorted(old-new),'gained':sorted(new-old)}
 old={r['object_id'] for r in old_ztf if r['own_error_deg'] is not None}
 new={r['object_id'] for r in ztf if r['own_error_deg'] is not None}
 return {'reliability_changes':changes,'ztf_changes':{'lost':sorted(old-new),'gained':sorted(new-old)},
         'membership_changed':bool(changes or old!=new),
         'note':'Searches are fixed; new changes in successful membership require inspection for prediction/fit failures.'}

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output', type=Path, required=True)
 args=parser.parse_args(argv)
 global O
 O=args.output
 network.guard_output(O)
 O.mkdir(parents=True,exist_ok=True)
 for name in ('reliability-paired.json','ztf-found-comparators.json','alcdef-random-unchanged.json','membership.json'):
  if (O/name).exists():raise FileExistsError('derived output already exists; verify the runner receipt')
 source=DataPath('failed-excluded-derived/derive.py')
 if network.sha256(source)!=SOURCE_SHA256:raise ValueError('published comparator producer changed')

 if network.switched():
  for path in (network.NetworkPath('results','ztf','own-period-scores.json'),network.NetworkPath('results','reliability','score/reliability-own-period.json')):network.check_json(path)

 ns={'np':np}
 load_functions(F/'source/repro/compute_reference_baselines.py', ['unit_from_degrees','standard_axes','axial_error_deg'],ns)
 ns['STANDARD_POLES_DEG']=((0,0),(180,0),(90,60),(240,60),(90,-60),(240,-60))
 cat={r['object_id']:r for r in map(json.loads,(F/'source/repro/data/damit-20250610T000301Z/catalog.jsonl').read_text().splitlines())}
 refs={i:np.array([s['vector'] for s in r['solutions'] if s.get('vector')]) for i,r in cat.items()}
 refs={i:v/np.linalg.norm(v,axis=1,keepdims=True) for i,v in refs.items() if v.size}
 base=np.load(F/'source/repro/baselines_out/reference-baselines-per-object.npz')
 std=dict(zip(base['object_ids'].astype(str),base['standard_errors_deg']))
 rand=dict(zip(base['object_ids'].astype(str),base['random_errors_deg']))
 other=np.load(F/'source-op-reliability/repro/baselines_out/reference-baselines-per-object.npz')
 assert set(other['object_ids'])==set(std)
 for i,v in zip(other['object_ids'],other['standard_errors_deg']):assert abs(std[str(i)]-v)<1e-10
 rows=read(network.NetworkPath('results','reliability','score/scored-rows-own.json',legacy_path=D/'reliability/score/scored-rows-own.json')); summary=read(network.NetworkPath('results','reliability','score/reliability-own-period.json',legacy_path=D/'reliability/score/reliability-own-period.json'))['failed_excluded']
 if network.switched() and any(any(r.get(k)!=v for k,v in network.json_metadata().items()) for r in rows):
  raise ValueError('reliability comparator rows belong to different networks')
 save('membership.json',membership_changes(read(D/'reliability/score/scored-rows-own.json'),rows,
      read(D/'own-period-scores.json')['ztf']['objects'],
      read(network.NetworkPath('results','ztf','own-period-scores.json',legacy_path=D/'own-period-scores.json'))['ztf']['objects']))
 conditions={}
 for name in summary['conditions']:
  cells=[r for r in rows if r['condition']==name and r['status']!='ineligible']
  good=[r for r in cells if r['status']=='ok']
  per=defaultdict(list)
  for r in good:per[r['object_id']].append(r['error_deg'])
  ids=sorted(per)
  result={'n_eligible':len({r['object_id'] for r in cells}),'n_objects':len(ids),'n_failed':len(cells)-len(good),'n_selections':len(cells),'n_found_selections':len(good)}
  if ids:
   de=np.array([np.mean(per[i]) for i in ids]);st=np.array([std[i] for i in ids]);diff=st-de
   draws=np.random.default_rng(20260930).integers(0,len(ids),size=(10000,len(ids)));boot=diff[draws].mean(axis=1)
   result.update(delphi_mean_deg=float(de.mean()),standard_mean_deg=float(st.mean()),random_mean_deg=float(np.mean([rand[i] for i in ids])),standard_minus_delphi={'mean_difference_deg':float(diff.mean()),'ci95_low_deg':float(np.percentile(boot,2.5)),'ci95_high_deg':float(np.percentile(boot,97.5))},objects=[{'object_id':i,'delphi_mean_deg':float(m),'standard_error_deg':float(std[i]),'successful_repeats':len(per[i])} for i,m in zip(ids,de)])
   assert abs(de.mean()-summary['conditions'][name]['mean_error_deg'])<1e-10
   if name.startswith('observation_cap'):
    old=summary['sampling_baselines']['total_cap_standard_minus_delphi_deg'][name.split('-')[-1]]
    assert np.allclose([len(ids),diff.mean(),np.percentile(boot,2.5),np.percentile(boot,97.5)],old,atol=1e-10)
   if name.startswith('two_d-block_count-') and name.endswith('cap-None'):
    key=name.split('-')[2]
    if key!='None':
     old=summary['sampling_baselines']['standard_minus_delphi_deg'][key]
     assert np.allclose([diff.mean(),np.percentile(boot,2.5),np.percentile(boot,97.5),len(ids)],old,atol=1e-10)
  conditions[name]=result
 # Reproduce the original random-axis draws on the same 80 identities, then pair
 # the fraction within 20 degrees on each repeat's successful identities.
 grid_ids=sorted(r['object_id'] for r in read(D/'reliability/score/scored-rows-own.json') if r['condition']=='grid-full' and r['status']=='ok')
 rng=np.random.default_rng(1);random_errors=np.empty((1000,len(grid_ids)))
 for k in range(1000):
  axes=rng.normal(size=(len(grid_ids),3,3));axes/=np.linalg.norm(axes,axis=2,keepdims=True)
  for j,i in enumerate(grid_ids):random_errors[k,j]=ns['axial_error_deg'](axes[j],refs[i])
 for groups in (1,3,5,10):
  name=f'two_d-block_count-{groups}-per_block_cap-None';c=conditions[name]
  ds=[];ss=[];rr=[];nn=[]
  for k in (0,1,2):
   good={r['object_id']:r for r in rows if r['condition']==name and r['repeat']==k and r['status']=='ok'}
   ids=sorted(good);nn.append(len(ids))
   ds.append(float(np.mean([good[i]['error_deg']<20 for i in ids])))
   ss.append(float(np.mean([std[i]<20 for i in ids])))
   rr.append(float(np.mean(random_errors[:,[grid_ids.index(i) for i in ids]]<20)))
  c.update(within20_delphi_by_repeat=ds,within20_standard_by_repeat=ss,within20_random_by_repeat=rr,n_found_by_repeat=nn,within20_delphi=float(np.mean(ds)),within20_standard=float(np.mean(ss)),within20_random=float(np.mean(rr)))
  assert abs(c['within20_delphi']-summary['sampling_baselines']['groups'][str(groups)]['within20_mean'])<1e-12
 save('reliability-paired.json',{'sources':['DeLPHI-followup/data/own-period-20261021/reliability/score/scored-rows-own.json','DeLPHI-followup/source/repro/baselines_out/reference-baselines-per-object.npz'],'method':'Mean successful-repeat error per asteroid. Standard errors on identical identities. Paired bootstrap over asteroids, 10000 resamples, seed 20260930. Within-20 fractions pair identities per successful repeat, then average three repeats. Random fractions reproduce sampling_baselines.py, 1000 draws, seed 1. All existing paired differences verified.','conditions':conditions})
 # ZTF tables need comparator summaries on the 108 successful identities.
 z=read(network.NetworkPath('results','ztf','own-period-scores.json',legacy_path=D/'own-period-scores.json'))['ztf'];ids=[r['object_id'] for r in z['objects']];mask=np.array([r['own_error_deg'] is not None for r in z['objects']]);keep=[i for i,m in zip(ids,mask) if m]
 no=read(D/'nettests/no-uncertainty/score-own.json')['variants']['ztf'];st={r['object_id']:r['standard_error_deg'] for r in no['objects']}
 fit={r['object_id']:r['train_only_atlas_error_deg'] for r in read(F/'data/publication-revision-20260917-r2/error-channel-sensitivity.json')['ztf']['objects']}
 # Reproduce own_period.score.external's 1000 draws for all 161 in the frozen order.
 rng=np.random.default_rng(20260922+len('ztf'));draw_errors=np.empty((1000,len(ids)))
 for k in range(1000):
  axes=rng.normal(size=(len(ids),3,3));axes/=np.linalg.norm(axes,axis=2,keepdims=True)
  for j,i in enumerate(ids):draw_errors[k,j]=ns['axial_error_deg'](axes[j],refs[i])
 def stats(e):return {'n':len(e),'mean_deg':float(np.mean(e)),'median_deg':float(np.median(e)),'within_20_percent':float(np.mean(np.array(e)<=20+1e-8)*100),'within_30_percent':float(np.mean(np.array(e)<=30+1e-8)*100)}
 rr=draw_errors[:,mask]
 zt={'n_total':z['n'],'n_failed':z['n_failed'],'n_found':len(keep),'standard':stats([st[i] for i in keep]),'fitted_axes':stats([fit[i] for i in keep]),'random':{'n':len(keep),'mean_deg':float(rr.mean()),'median_deg':float(np.median(rr,axis=1).mean()),'within_20_percent':float((rr<=20+1e-8).mean()*100),'within_30_percent':float((rr<=30+1e-8).mean()*100)},'object_ids':keep,'sources':['DeLPHI-followup/data/own-period-20261021/own-period-scores.json :: ztf/objects','DeLPHI-followup/data/own-period-20261021/nettests/no-uncertainty/score-own.json :: variants/ztf/objects','DeLPHI-followup/data/publication-revision-20260917-r2/error-channel-sensitivity.json :: ztf/objects'],'random_method':'1000 draws, seed 20260925, all 161 identities in own-period-scores.json order, then select same 108 successful identities. Median and hit fractions averaged across random draws.'}
 for name in ['standard','fitted_axes','random']:
  assert abs(zt[name]['mean_deg']-z['own_period_failed_excluded']['mean_deg']-z[name+'_minus_own_failed_excluded']['mean_difference_deg'])<1e-10
 save('ztf-found-comparators.json',zt)
 print('PASS: existing reduced-data paired means and intervals reproduced; ZTF comparator means reproduce stored paired differences')
 for g in (1,3,5,10):
  c=conditions[f'two_d-block_count-{g}-per_block_cap-None'];print('groups',g,{k:c[k] for k in ['n_failed','n_objects','delphi_mean_deg','standard_mean_deg','within20_delphi','within20_standard','within20_random']})
 for k in ['standard','random','fitted_axes']:print('ZTF',k,zt[k])
 
 shutil.copyfile(D/'failed-excluded-derived/alcdef-random-unchanged.json',O/'alcdef-random-unchanged.json')


if __name__ == '__main__':
 main()
