"""Deterministic exact engine token IDs; paired content is allocated before timing."""
import random
import re
from pathlib import Path
from .contracts import digest, save


def materialize(cases, order, tokenizer, *, model, revision, tokenizer_sha256):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Pin the model/tokenizer to a full immutable HF commit')
    vocab = sorted(set(tokenizer.get_vocab().values())-set(tokenizer.all_special_ids))
    if len(vocab) < 2:
        raise ValueError('No ordinary tokenizer vocabulary')
    by_case = {c['id']:c for c in cases}
    pools, trials = {}, {}
    for trial in order:
        c = by_case[trial['case']]
        # Same content pools across cases within a block and within an off/light pair.
        pool = (f"pair:{trial['paired_content_key']}" if 'paired_content_key' in trial
                else f"block:{trial['block']}" if 'block' in trial else trial['stage'])
        requests = []
        specs = [(f'background-{i}',1024,256,0.,'background') for i in range(c['background_decoders'])]
        specs += [(f'foreground-{i}',n,128,offset/1000.,'foreground') for i,(n,offset) in enumerate(zip(c['foreground_tokens'],c['foreground_offsets_ms']))]
        if c['sentinel']:
            specs.append(('sentinel',512,128,c['sentinel']['arrival_ms']/1000.,'sentinel'))
        for rid,n,out,at,role in specs:
            key = f'{pool}:{rid}:{n}'
            if key not in pools:
                rng = random.Random(int(digest(['prefill-content',20261024,pool,rid]),16))
                ids = [rng.choice(vocab) for _ in range(n)]
                pools[key] = dict(input_ids=ids, sha256=digest(ids))
            requests.append(dict(id=rid,token_key=key,input_tokens=n,max_tokens=out,at=at,role=role))
        trials[trial['trial_id']] = requests
    manifest = dict(schema='prefill-token-manifest-v1', model=model, revision=revision,
                    tokenizer_sha256=tokenizer_sha256, content_seed=20261024,
                    cases_sha256=digest(cases), order_sha256=digest(order),
                    pools=pools,trials=trials, forced_outputs=True, ignore_eos=True)
    validate(manifest,cases,order)
    return manifest


def validate(manifest, cases, order):
    if manifest['cases_sha256'] != digest(cases) or manifest['order_sha256'] != digest(order):
        raise ValueError('Token manifest does not match frozen cases/order')
    if set(manifest['trials']) != {t['trial_id'] for t in order}:
        raise ValueError('Missing/extra token trial')
    for key,pool in manifest['pools'].items():
        ids = pool['input_ids']
        if not ids or any(type(n) is not int or n<0 for n in ids) or digest(ids) != pool['sha256']:
            raise ValueError('Invalid/changed token pool: '+key)
    by_case = {c['id']:c for c in cases}
    for t in order:
        requests = manifest['trials'][t['trial_id']]
        c = by_case[t['case']]
        expected = {f'background-{i}':(1024,256,0.,'background') for i in range(c['background_decoders'])}
        expected.update({f'foreground-{i}':(n,128,a/1000.,'foreground') for i,(n,a) in enumerate(zip(c['foreground_tokens'],c['foreground_offsets_ms']))})
        if c['sentinel']:expected['sentinel']=(512,128,c['sentinel']['arrival_ms']/1000.,'sentinel')
        if len(requests)!=len(expected) or {r['id'] for r in requests}!=set(expected):
            raise ValueError('Wrong request count/identities')
        for r in requests:
            if (r['input_tokens'],r['max_tokens'],r['at'],r['role']) != expected[r['id']]:
                raise ValueError('Token request differs from frozen cell')
            if len(manifest['pools'][r['token_key']]['input_ids'])!=r['input_tokens'] or r['input_tokens']+r['max_tokens']>8192:
                raise ValueError('Wrong exact input count or context overflow')
    return digest(manifest)


def peak_kv_tokens(requests,block_size):
    if type(block_size) is not int or block_size<=0:raise ValueError('Invalid KV block size')
    return sum(((r['input_tokens']+r['max_tokens']+block_size-1)//block_size)*block_size for r in requests)
