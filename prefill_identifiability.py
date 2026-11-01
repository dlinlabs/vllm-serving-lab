"""Training-only structural checks for the proposed service models, CPU only."""
import numpy as np

FEATURES=('intercept','decode_tokens','prefill_tokens','decode_context','attention_pairs')
FAMILIES={'M0':3,'M1':4,'M2':5}
CONDITION_LIMIT=1e4  # engineering guard after training-only column normalization


def features(step):
    z=sum(q*l+q*(q+1)/2 for _,q,l in step['chunks'])
    return [1.,step['decode'],step['prefill'],step['decode_context'],z]


def matrix_diagnostics(rows,columns):
    a=np.asarray(rows,dtype=float)
    if a.ndim!=2 or a.shape[1]!=len(FEATURES) or not len(a) or not np.isfinite(a).all():
        raise ValueError('Expected nonempty finite five-feature rows')
    a=a[:,:columns]
    norms=np.linalg.norm(a,axis=0)
    scaled=a/np.where(norms>0,norms,1.)
    rank=int(np.linalg.matrix_rank(scaled))
    condition=float(np.linalg.cond(scaled)) if rank==columns else None
    return dict(rows=len(a),features=list(FEATURES[:columns]),rank=rank,
        zero_columns=[FEATURES[i] for i,n in enumerate(norms) if n==0],
        training_column_norms=norms.tolist(),scaled_condition_number=condition,
        full_column_rank=rank==columns,
        supported=rank==columns and condition is not None and condition<=CONDITION_LIMIT)


def calibration_support(case_rows):
    """One structural scenario per calibration cell, with case-balanced row weights.

    Every LOCO matrix and its normalization exclude the held cell. Real fitting
    must repeat these checks on valid training observations, not reuse toy ranks.
    """
    if len(case_rows)<2:raise ValueError('Need multiple calibration cells')
    groups={k:np.asarray(v,float) for k,v in case_rows.items()}
    if any(v.ndim!=2 or v.shape[1]!=5 or not len(v) or not np.isfinite(v).all() for v in groups.values()):
        raise ValueError('Invalid calibration rows')
    out=[]
    for hold in (None,*groups):
        names=[k for k in groups if k!=hold]
        rows=np.concatenate([groups[k]/np.sqrt(len(groups[k])) for k in names])
        out.append(dict(held_cell=hold,training_cells=names,
            models={m:matrix_diagnostics(rows,n) for m,n in FAMILIES.items()}))
    return dict(role='toy structural support, not fitted parameter certainty or measured validation',
        row_scope='foreground release through last foreground completion; background prep/drain excluded',
        weighting='each calibration cell has equal total squared row weight',
        condition_limit=CONDITION_LIMIT,folds=out,
        all_families_supported=all(v['supported'] for f in out for v in f['models'].values()))
