"""Calibration contract for exclusive stage service; native labels not qualified yet."""
import copy
import numpy as np
from scipy.optimize import nnls
from .async_predictor import FEATURES, LAUNCHES, PARTS, SCALARS, PipelineCosts
from .contracts import digest

CHANNELS = tuple(f'{s}.{p}' for s in LAUNCHES for p in PARTS) + SCALARS


def plan():
    return dict(schema='async-calibration-plan-v1', status='cpu_contract_only',
        native_collection_ready=False, condition_limit=1000.,
        trials={**{f'train-{i}':'train' for i in range(6)},
                **{f'held-{i}':'held' for i in range(3)}},
        laws={c:(['bias','prefill','decode_context','attention_pairs']
                 if c=='forward.gpu_tail' else ['bias','num_requests']) for c in CHANNELS},
        label_scope='exclusive_service_s',
        native_blocker='First/last enqueue and wait-exclusive service extraction, clock/stream scope and observer overhead require qualification')


def validate_plan(p):
    if p['schema'] != 'async-calibration-plan-v1' or set(p['laws']) != set(CHANNELS):
        raise ValueError('Invalid plan channels/schema')
    if not p['trials'] or set(p['trials'].values()) != {'train','held'}:
        raise ValueError('Plan requires disjoint train/held trial IDs')
    if p['label_scope'] != 'exclusive_service_s':
        raise ValueError('Wait-inclusive or nested timing labels are not service costs')
    if type(p['condition_limit']) not in (int,float) or not np.isfinite(p['condition_limit']) or not 1 <= p['condition_limit'] <= 1000:
        raise ValueError('Invalid condition limit')
    for names in p['laws'].values():
        if not names or len(set(names)) != len(names) or set(names)-FEATURES:
            raise ValueError('Invalid fixed feature basis')


def rows_for(data, p, role):
    validate_plan(p)
    if data['schema'] != 'async-service-labels-v1' or data['plan_sha256'] != digest(p):
        raise ValueError('Dataset/plan identity mismatch')
    if data['origin'] != 'synthetic':
        raise ValueError('Native calibration blocked: service-label qualification is not implemented')
    if not isinstance(data['runtime_sha256'],str) or len(data['runtime_sha256']) != 64 or any(c not in '0123456789abcdef' for c in data['runtime_sha256']):
        raise ValueError('Runtime identity required')
    selected=[]; seen=set()
    for row in data['rows']:
        tid=row['trial_id']
        if tid not in p['trials']:
            raise ValueError('Unknown trial ID')
        # Do not even inspect features/labels of the other split.
        if p['trials'][tid] != role:
            continue
        key=(tid,row['step_id'])
        if key in seen:raise ValueError('Duplicate trial/step')
        seen.add(key)
        if not isinstance(row['step_id'],str) or not row['step_id']:
            raise ValueError('Stable step identity required')
        if row['mode'] not in ('NONE','FULL','PIECEWISE','EMPTY'):
            raise ValueError('Unknown mode')
        if set(row['features']) != FEATURES-{'bias'} or set(row['labels']) != set(CHANNELS):
            raise ValueError('Missing/extra features or component labels')
        for v in row['features'].values():
            if type(v) not in (int,float) or not np.isfinite(v) or v<0:raise ValueError('Invalid feature')
        for label in row['labels'].values():
            if label['scope'] != p['label_scope']:raise ValueError('Wait-inclusive or nested label rejected')
            v=label['seconds']
            if type(v) not in (int,float) or not np.isfinite(v) or v<0:raise ValueError('Invalid service label')
        selected.append(row)
    if {r['trial_id'] for r in selected} != {k for k,v in p['trials'].items() if v==role}:
        raise ValueError('Incomplete declared trial split')
    return sorted(selected,key=lambda r:(r['trial_id'],r['step_id']))


def matrix(rows,names):
    return np.asarray([[1. if k=='bias' else r['features'][k] for k in names] for r in rows])


def identify(x, weights, limit):
    scale=np.sqrt(np.average(x*x,axis=0,weights=weights))
    if np.any(scale==0):raise ValueError('Unidentifiable zero feature')
    z=x/scale
    weighted=z*np.sqrt(weights)[:,None]
    rank=int(np.linalg.matrix_rank(weighted));condition=float(np.linalg.cond(weighted))
    if rank != x.shape[1] or not np.isfinite(condition) or condition>limit:
        raise ValueError('Unidentifiable or ill-conditioned service law')
    return weighted,scale,dict(rank=rank,condition=condition,scale=scale.tolist())


def freeze(data,p):
    rows=rows_for(data,p,'train')
    counts={t:sum(r['trial_id']==t for r in rows) for t in p['trials'] if p['trials'][t]=='train'}
    weights=np.array([1./counts[r['trial_id']] for r in rows])
    models={}; diagnostics={}; ranges={}
    for channel,names in p['laws'].items():
        x=matrix(rows,names);y=np.array([r['labels'][channel]['seconds'] for r in rows])
        z,scale,info=identify(x,weights,p['condition_limit'])
        coef,_=nnls(z,y*np.sqrt(weights));coef=coef/scale
        models[channel]=dict(zip(names,map(float,coef)));diagnostics[channel]=info
        ranges[channel]=dict(min=x.min(axis=0).tolist(),max=x.max(axis=0).tolist())
    costs=dict(schema='async-stage-costs-v1',provenance='Synthetic calibration only; not a native physical model',
        launches={s:{part:models[f'{s}.{part}'] for part in PARTS} for s in LAUNCHES},
        scalar={s:models[s] for s in SCALARS})
    PipelineCosts.from_dict(costs)
    artifact=dict(schema='async-calibration-freeze-v1',plan=copy.deepcopy(p),plan_sha256=digest(p),
        runtime_sha256=data['runtime_sha256'],training_sha256=digest(rows),
        training_trials=sorted(counts),training_modes=sorted({r['mode'] for r in rows}),
        diagnostics=diagnostics,ranges=ranges,costs=costs,model_accepted=False,
        native_collection_ready=False,status='synthetic_fit_only')
    artifact['freeze_sha256']=digest(artifact)
    return artifact


def evaluate(data,frozen):
    payload={k:v for k,v in frozen.items() if k!='freeze_sha256'}
    if digest(payload)!=frozen['freeze_sha256']:raise ValueError('Freeze integrity mismatch')
    p=frozen['plan']
    if digest(p)!=frozen['plan_sha256']:raise ValueError('Frozen plan mismatch')
    if data['runtime_sha256']!=frozen['runtime_sha256']:raise ValueError('Runtime changed')
    rows=rows_for(data,p,'held');costs=PipelineCosts.from_dict(frozen['costs']);scores={}
    modes=sorted({r['mode'] for r in rows}-set(frozen['training_modes']))
    supported=not modes
    for channel,names in p['laws'].items():
        law=costs.launches[channel.split('.')[0]][channel.split('.')[1]] if '.' in channel else costs.scalar[channel]
        x=matrix(rows,names);bounds=frozen['ranges'][channel]
        outside=np.any((x<np.array(bounds['min']))|(x>np.array(bounds['max'])),axis=1)
        supported=supported and not bool(np.any(outside))
        errors=[abs(law(r['features'])-r['labels'][channel]['seconds']) for r in rows]
        means={tid:float(np.mean([e for r,e in zip(rows,errors) if r['trial_id']==tid]))
               for tid in sorted({r['trial_id'] for r in rows})}
        scores[channel]=dict(trial_mae_s=means,balanced_mae_s=float(np.mean(list(means.values()))),
                             out_of_range_rows=int(sum(outside)))
    return dict(schema='async-calibration-held-v1',freeze_sha256=frozen['freeze_sha256'],
        held_sha256=digest(rows),scores=scores,unseen_modes=modes,within_marginal_support=supported,
        model_accepted=False,status='synthetic_evaluation_only',
        limitation='Marginal feature bounds are not joint-support or latency qualification; no end-to-end TTFT acceptance')
