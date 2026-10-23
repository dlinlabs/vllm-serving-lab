"""Freeze actual V2 graph candidates and dispatch using only predicted work.

Compatibility mirrors the hash-pinned upstream CudaGraphManager._is_compatible.
No GPU calls, observed future mode tags or timing labels enter prediction.
"""


def freeze_v2_manager(manager):
    candidates={}
    for (tokens,loras),rows in manager._candidates.items():
        if loras:continue
        candidates[str(tokens)]=[dict(mode=d.cg_mode.name,num_tokens=d.num_tokens,num_reqs=d.num_reqs,
                                     uniform_token_count=d.uniform_token_count,max_query_len=d.max_query_len,
                                     num_active_loras=d.num_active_loras,num_ubatches=d.num_ubatches) for d in rows]
    return dict(schema='v2-dispatch-candidates-v1',graphs_captured=manager._graphs_captured,candidates=candidates)


def v2_dispatch(frozen,num_reqs,num_tokens,max_query_len,has_prefill):
    uniform=None if has_prefill or num_tokens!=num_reqs*max_query_len else max_query_len
    if frozen['graphs_captured']:
        for d in frozen['candidates'].get(str(num_tokens),[]):
            if ((d['uniform_token_count'] is None or d['uniform_token_count']==uniform)
                and (d['max_query_len'] is None or d['max_query_len']>=max_query_len)
                and (d['num_reqs'] is None or d['num_reqs']>=num_reqs)
                and d['num_tokens']>=num_tokens and d['num_active_loras']==0 and d['num_ubatches']==1):
                return d['mode'],d['num_tokens'],d['num_reqs'],d['uniform_token_count'] is not None
    return 'NONE',num_tokens,num_reqs,False
