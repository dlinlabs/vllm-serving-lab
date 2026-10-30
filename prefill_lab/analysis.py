"""Calibration-only fitting and integrity gates; held timings never select models."""
from collections import defaultdict
import math
import numpy as np
from scipy.optimize import nnls
from scipy.stats import t
from .contracts import digest,finite
from prefill_grouped_validation import calibration_noise_readiness

FAMILIES={'M0':('decode','prefill'),'M1':('decode','prefill','decode_context'),
          'M0Z':('decode','prefill','attention_pairs'),'M2':('decode','prefill','decode_context','attention_pairs')}


def observer_gate(captures,order):
    by_id={c['trial_id']:c for c in captures}
    results=[]
    for case,n in (('single-L1024-D0',20),('composition-uniform-D16',10)):
        pairs=defaultdict(dict)
        for trial in order:
            if trial['stage']=='observer_check' and trial['case']==case:
                c=by_id.get(trial['trial_id'])
                if c is None or c['status']!='complete' or c.get('errors'):
                    return dict(status='inconclusive',reason='Missing/invalid observer trial')
                foreground=[r['first']-c['release'] for r in c['requests'].values() if r['role']=='foreground']
                pairs[trial['pair']][trial['observer']]=(c['end']-c['start'],max(foreground))
        if len(pairs)!=n or any(set(p)!= {'off','light'} for p in pairs.values()):
            return dict(status='inconclusive',reason='Incomplete observer pairs')
        for index,target in enumerate(('whole_trial','foreground_ttft')):
            ratios=np.array([math.log(p['light'][index]/p['off'][index]) for p in pairs.values()])
            upper=math.expm1(float(ratios.mean()+t.ppf(.95,n-1)*ratios.std(ddof=1)/math.sqrt(n)))
            results.append(dict(case=case,target=target,log_ratios=ratios.tolist(),upper_fraction=upper,passed=upper<=.05))
    return dict(status='passed' if all(r['passed'] for r in results) else 'failed',rows=results,
                interpretation='Geometric-mean engineering overhead allowance, not zero observer effect')


def rows_from_capture(capture,label='wall'):
    if capture['status']!='complete' or capture.get('errors') or capture.get('overflow'):
        raise ValueError('Cannot fit invalid capture')
    if capture['release'] is None or capture['foreground_done'] is None:
        raise ValueError('Missing foreground step window')
    workers={r['step_id']:r for w in capture['workers'] for r in w['rows']}
    rows=[]
    for step in capture['steps']:
        if step['start']<capture['release'] or step['end']>capture['foreground_done']:continue
        worker=workers[step['step_id']]
        if worker['runtime_mode'] not in ('NONE','PIECEWISE','FULL') or worker['padded_tokens'] is None:
            raise ValueError('Unavailable runtime mode/padding')
        if worker.get('microbatching'):raise ValueError('Microbatch spans require another reviewed model')
        if label=='gpu' and not capture.get('cuda_scope_qualified',False):
            raise ValueError('GPU forward scope not qualified by actual deep traces')
        y=step['end']-step['start'] if label=='wall' else worker['gpu_forward_s']
        if not finite(y) or y<=0:raise ValueError('Invalid timing label')
        rows.append(dict(**{k:step[k] for k in ('decode','prefill','decode_context','attention_pairs')},
                         mode=worker['runtime_mode'],padding_excess=worker['padded_tokens']-step['decode']-step['prefill'],
                         y=y,trial_id=capture['trial_id'],step_id=step['step_id'],start=step['start'],end=step['end'],
                         work=step['work'],padded_tokens=worker['padded_tokens']))
    if not rows:raise ValueError('Empty scored service window')
    return rows


def basis(training):
    rows=[r for cell in training.values() for r in cell]
    modes=sorted({r['mode'] for r in rows})
    # Frozen rule: omit a globally constant padding column, never inspect held rows.
    padding=len({r['padding_excess'] for r in rows})>1
    return dict(modes=modes,padding=padding,padding_constant=None if padding else rows[0]['padding_excess'],features={f:['mode:'+m for m in modes]+list(cols)+(['padding_excess'] if padding else []) for f,cols in FAMILIES.items()})


def vector(row,names):
    if 'mode:'+row['mode'] not in names:raise ValueError('Mode absent from calibration training support')
    return [float(row['mode']==n[5:]) if n.startswith('mode:') else float(row[n]) for n in names]


def fit(training,family,encoding):
    names=encoding['features'][family];x=[];y=[]
    for rows in training.values():
        weight=1/math.sqrt(len(rows))
        x.extend(np.array([vector(r,names) for r in rows])*weight)
        y.extend(r['y']*weight for r in rows)
    x,y=np.asarray(x),np.asarray(y);scale=np.linalg.norm(x,axis=0)
    if (scale==0).any():raise ValueError('Unsupported zero feature column')
    normalized=x/scale;condition=float(np.linalg.cond(normalized))
    if np.linalg.matrix_rank(normalized)!=len(names) or condition>1e4:
        raise ValueError('Unsupported mode-aware rank/condition')
    return dict(features=names,padding_constant=encoding['padding_constant'],coefficients=(nnls(normalized,y)[0]/scale).tolist(),
                training_column_norms=scale.tolist(),rank=len(names),condition=condition)


def predict(row,model):
    if model.get('padding_constant') is not None and row['padding_excess']!=model['padding_constant']:
        raise ValueError('Padding absent from training support')
    return float(np.dot(vector(row,model['features']),model['coefficients']))


def errors(rows,model):
    # A cell contains independent trials. Reset all cumulative sums at each trial.
    trials=defaultdict(list)
    for row in rows:trials[row['trial_id']].append(row)
    scores=[]
    for trial_rows in trials.values():
        residual=np.array([predict(r,model)-r['y'] for r in trial_rows])
        p=np.array([r['prefill']>0 for r in trial_rows])
        if not p.any() or p.all():raise ValueError('Need both prefill and decode observations per trial')
        endpoints={}
        for i,row in enumerate(trial_rows):
            for w in row['work']:
                if w['position']<w['prompt_length'] and w['request_id'].split('/')[-1] in ('sentinel','foreground-0','foreground-1','foreground-2','foreground-3'):
                    endpoints[w['request_id']]=i
        if not endpoints:raise ValueError('No foreground cumulative targets')
        scores.append(dict(decode=float(abs(residual[~p]).mean()),prefill=float(abs(residual[p]).mean()),
                           cumulative=float(np.mean([abs(residual[:i+1].sum()) for i in endpoints.values()]))))
    return {target:float(np.mean([s[target] for s in scores])) for target in ('decode','prefill','cumulative')}


def calibration_readiness(training,order):
    """Twelve fixed cell/phase totals; never maximize over hundreds of step CVs.

    Compare complete ordered work/mode/padding signatures across three trials
    before pooling durations. This is phase-total repeatability, not step-label CV.
    """
    from scripts.build_prefill_design import cases
    expected=sorted(c['id'] for c in cases() if c['split']=='calibration')
    declared={t['trial_id']:t for t in order if t.get('analysis_role')=='train'}
    durations={};signatures={};step_cvs=[]
    for case in expected:
        trials=defaultdict(list)
        for row in training.get(case,[]):trials[declared[row['trial_id']]['block']].append(row)
        for phase in ('prefill','decode'):
            key=case+'/'+phase;values=[];identities=[];step_values=[]
            for block in (1,2,3):
                rows=[r for r in trials[block] if (r['prefill']>0)==(phase=='prefill')]
                identities.append([(r['mode'],r['padded_tokens'],[(w['request_id'].split('/')[-1],w['position'],w['tokens']) for w in r['work']]) for r in rows])
                values.append(sum(r['y'] for r in rows));step_values.append([r['y'] for r in rows])
            signatures[key]=digest(identities[0])
            if not identities[0] or identities[1:]!=identities[:1]*2:
                return dict(status='inconclusive',reason='Missing or unmatched fixed cell/phase work',model_accepted=False)
            durations[key]=values
            array=np.array(step_values)
            step_cvs.extend((array.std(axis=0,ddof=1)/array.mean(axis=0)).tolist())
    result=calibration_noise_readiness(durations,expected_strata=[c+'/'+p for c in expected for p in ('prefill','decode')])
    result.update(protocol='six-cells-two-phase-totals-v1',stratum_count=12,work_signature_sha256=signatures,
                  phase_totals_s=durations,descriptive_max_step_cv=max(step_cvs),
                  step_cv_controls_gate=False,
                  note='Fixed phase-total CV screen; not a population bound, per-step noise estimate or power guarantee. Per-step maximum is descriptive only.')
    return result


def freeze_models(captures,order,label='wall'):
    allowed={t['trial_id']:t for t in order if t.get('analysis_role')=='train'}
    # All other roles, including repeat_check and held timings, are excluded BEFORE reading labels.
    chosen=[c for c in captures if c['trial_id'] in allowed]
    if len(chosen)!=18 or len({c['trial_id'] for c in chosen})!=18:
        raise ValueError('Need every one of the18 frozen calibration trials')
    training=defaultdict(list)
    for c in chosen:
        trial=allowed[c['trial_id']]
        rows=rows_from_capture(c,label)
        if any(b['start']<a['end'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Overlapping wall labels cannot support serial cumulative selection')
        training[trial['case']].extend(rows)
    readiness=calibration_readiness(training,order)
    encoding=basis(training);fits={};scores={};unsupported={}
    for family in FAMILIES:
        try:
            folds={}
            for held,rows in training.items():
                part={k:v for k,v in training.items() if k!=held}
                # Fold vocabularies, constant-column rules and scales use only that fold's training cells.
                folds[held]=errors(rows,fit(part,family,basis(part)))
            scores[family]=folds;fits[family]=fit(training,family,encoding)
        except ValueError as exc:unsupported[family]=str(exc)
    if 'M0' not in fits:raise ValueError('Mode-aware M0 lacks training support')
    def retain(candidate,base,primary,guards):
        if candidate not in fits:return False
        for target in primary:
            a=np.mean([v[target] for v in scores[candidate].values()]);b=np.mean([v[target] for v in scores[base].values()])
            if b<=1e-12 or b-a<.1*b:return False
        return all(scores[candidate][k][target]-v<=max(.1*v,.001) for k,row in scores[base].items() for target,v in row.items() if target in guards)
    base='M1' if retain('M1','M0',('decode',),('prefill','cumulative')) else 'M0'
    selected='M2' if retain('M2',base,('prefill','cumulative'),('decode','prefill','cumulative')) else base
    plain_training={k:[dict(r,mode='PLAIN') for r in rows] for k,rows in training.items()}
    plain_encoding=dict(padding_constant=None,features={'M0':['mode:PLAIN','decode','prefill']})
    try:
        plain_fit=fit(plain_training,'M0',plain_encoding)
        plain_loco={held:errors(rows,fit({k:v for k,v in plain_training.items() if k!=held},'M0',plain_encoding)) for held,rows in plain_training.items()}
        plain_ablation=dict(fit=plain_fit,loco_errors=plain_loco,selectable=False)
    except ValueError as exc:plain_ablation=dict(unsupported=str(exc),selectable=False)
    context_strata=defaultdict(list)
    for case,rows in training.items():
        if case.endswith('-D16'):
            for r in rows:context_strata[(r['mode'],r['padded_tokens'])].append(r)
    context_support=[]
    for (mode,pad),rows in context_strata.items():
        x=np.array([[1,r['decode'],r['prefill']] for r in rows],float)
        y=np.array([r['decode_context'] for r in rows],float)
        residual=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
        context_support.append(dict(mode=mode,padded_tokens=pad,rows=len(rows),
                                    context_variation=float(np.ptp(y)),
                                    residual_context_norm=float(np.linalg.norm(residual)),
                                    noncollinear_context=bool(np.linalg.norm(residual)>1e-8)))
    return dict(schema='prefill-measured-model-freeze-v2',label=label,selected=selected,retained_base=base,
                encoding=encoding,fits=fits,loco_errors=scores,unsupported=unsupported,noise_readiness=readiness,plain_M0_ablation=plain_ablation,
                context_strata_support=context_support,unique_context_mechanism_claimed=False,
                training_capture_sha256={c['trial_id']:digest(c) for c in chosen},
                held_labels_used=False,model_accepted=False,block_dependence_status='unresolved')


def drift_report(captures,order):
    lookup={c['trial_id']:c for c in captures};pairs=defaultdict(dict)
    for t0 in order:
        if t0['stage']=='drift_reference' and t0['trial_id'] in lookup:
            c=lookup[t0['trial_id']];pairs[t0['block']][t0['position']]=c
    rows=[]
    for block,p in pairs.items():
        if set(p)!= {'start','end'}:
            rows.append(dict(block=block,status='incomplete'));continue
        a,b=p['start'],p['end'];ratio=(b['end']-b['start'])/(a['end']-a['start'])
        ttft=lambda c:max(r['first']-c['release'] for r in c['requests'].values() if r['role']=='foreground')
        tr=ttft(b)/ttft(a)
        rows.append(dict(block=block,wall_ratio=ratio,ttft_ratio=tr,flag=abs(ratio-1)>.1 or abs(tr-1)>.1))
    return dict(rows=rows,unqualified_pooled_claim_allowed=len(rows)==5 and all(not r.get('flag',True) for r in rows))
