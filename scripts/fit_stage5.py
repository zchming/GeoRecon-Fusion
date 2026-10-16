"""Fit only six priority permutations on DEV; deny test/mixed evaluation file reads."""
from itertools import permutations
from pathlib import Path
import sys
import time
from georecon.data.protocol import read_json,write_json,sha256
from georecon.evaluation.poses import evaluate_poses
from georecon.selection.policy import choose
ROOT=Path(__file__).resolve().parents[1];DOC=ROOT/'docs/stage5'


def dev_only_guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        filename=str(args[0]).replace('\\','/').lower()
        if any(s in filename for s in ['dtu_scan6','/scan6','/test/','evaluation.json','results.json']):
            raise PermissionError('test or mixed evaluation access forbidden during dev-only policy fit')


def score(evaluation,threshold=10):
    values=[]
    for pair in evaluation['pairs']:
        a,b=pair['rotation_deg'],pair['translation_direction_deg']
        values.append(max(0,1-max(a,b)/threshold) if a is not None and b is not None else 0.)
    return sum(values)/len(values)


def fit():
    start=time.perf_counter();reads=[]
    def guard(event,args):
        dev_only_guard(event,args)
        if event=='open' and isinstance(args[0],str) and args[0].startswith(str(ROOT)):
            reads.append(str(Path(args[0]).relative_to(ROOT)))
    sys.addaudithook(guard)
    if (ROOT/'configs/selection/stage5_v1.json').exists():raise ValueError('fitted policy already exists; never silently refit')
    protocol=read_json(ROOT/'configs/selection/stage5_protocol.json');lock=read_json(DOC/'input_lock.json')
    freeze=read_json(ROOT/'docs/stage3/freeze.json')
    assert lock['protocol_sha256']==sha256(ROOT/'configs/selection/stage5_protocol.json')
    dev=[]
    for case in lock['cases']:
        if case['split']!='dev':continue
        bundle=ROOT/case['bundle'];manifest=read_json(bundle/'manifest.json')
        assert manifest['split']=='dev' and manifest['source']['scan']==1
        predictions={m:read_json(bundle/(m+'.json')) for m in protocol['methods']}
        hashes={m:sha256(bundle/(m+'.json')) for m in protocol['methods']}
        diagnostics=read_json(bundle/'reliability.json')
        entry=next(d for d in freeze['datasets'] if d['manifest']['scene_id']==case['scene_id'])
        truth_path=(ROOT/entry['path']).parent/'reference.json';assert sha256(truth_path)==entry['reference_sha256']
        evaluations={m:evaluate_poses(p,read_json(truth_path)) for m,p in predictions.items()}
        dev.append(dict(scene_id=case['scene_id'],manifest=manifest,hashes=hashes,predictions=predictions,diagnostics=diagnostics,evaluations=evaluations))
    trials=[]
    for priority in permutations(protocol['methods']):
        policy=dict(schema_version=1,priority=list(priority),gate='complete_cameras_and_nonempty_evidence',partial_evidence='allow_with_warning',no_candidate='abstain')
        cases=[]
        for case in dev:
            manifest=case['manifest']
            decision=choose(case['predictions'],case['diagnostics'],policy,case['scene_id'],case['diagnostics']['input_manifest_sha256'],[i['image_id'] for i in manifest['images']],case['hashes'])
            selected=decision['selected_method']
            cases.append({'scene_id':case['scene_id'],'selected_method':selected,'joint_auc10':score(case['evaluations'][selected]) if selected else 0.,'fallback_used':decision['fallback_used']})
        trials.append({'policy':policy,'cases':cases,'mean_dev_joint_auc10':sum(c['joint_auc10'] for c in cases)/len(cases)})
    canonical={m:i for i,m in enumerate(protocol['canonical_tie_order'])}
    best=sorted(trials,key=lambda t:(-t['mean_dev_joint_auc10'],tuple(canonical[m] for m in t['policy']['priority'])))[0]
    selected=best['policy']
    selected.update(fit_protocol_sha256=sha256(ROOT/'configs/selection/stage5_protocol.json'),dev_scene_ids=[d['scene_id'] for d in dev],training_split='dev_only')
    write_json(ROOT/'configs/selection/stage5_v1.json',selected)
    write_json(DOC/'fit_report.json',{'objective':protocol['objective'],'trials':trials,'winning_priority':selected['priority'],'winning_mean_dev_joint_auc10':best['mean_dev_joint_auc10'],
                                    'policy_sha256':sha256(ROOT/'configs/selection/stage5_v1.json'),'input_lock_sha256':sha256(DOC/'input_lock.json'),
                                    'test_and_mixed_evaluation_reads_forbidden':True,'project_file_accesses':sorted(set(reads)),
                                    'fallback_identifiability':'winner always selects raw on actual dev; secondary order resolved by predeclared canonical tie-break, not learned from natural raw failures',
                                    'wall_seconds':time.perf_counter()-start})
    print('Frozen dev-only priority:',selected['priority'])


if __name__=='__main__':fit()
