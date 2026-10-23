"""Authenticated opt-in routes on vLLM's existing API server and utility channel."""
import hmac
import json
import os
from pathlib import Path

from .contracts import digest
from .tokens import validate


class Endpoint:
    name='prefill_lab'
    required_tasks=('generate',)

    def attach_router(self,app):
        from fastapi import HTTPException,Request
        # FastAPI resolves this annotation in the module namespace.
        globals()['Request']=Request
        @app.post('/prefill-lab/command')
        async def execute(request:Request):
            secret=os.environ.get('PREFILL_LAB_KEY','')
            if len(secret)<32 or not hmac.compare_digest(request.headers.get('authorization',''),'Bearer '+secret):
                raise HTTPException(401,'Invalid experiment credential')
            payload=await request.json()
            if not isinstance(payload,dict) or set(payload)-{'operation','trial_id','reason','observer'}:
                raise HTTPException(400,'Unexpected command fields')
            operation=payload.get('operation')
            if operation not in ('meta','start','status','abort','finish','external_begin','external_end'):
                raise HTTPException(400,'Unknown experiment operation')
            if operation=='start':
                tid=payload.get('trial_id')
                if tid not in self.trials:raise HTTPException(400,'Trial is not frozen')
                trial=dict(self.trials[tid])
                trial['requests']=[dict(r,input_ids=self.manifest['pools'][r['token_key']]['input_ids'])
                                   for r in self.manifest['trials'][tid]]
                payload=dict(operation='start',trial=trial)
            if operation=='external_begin':
                if payload.get('trial_id') not in self.external_ids or payload.get('observer','light')!='light':
                    raise HTTPException(400,'External stage is not frozen')
                if payload['trial_id'] in self.manifest['trials']:
                    payload=dict(payload,expected_foreground=sum(r['role']!='background' for r in self.manifest['trials'][payload['trial_id']]))
            result=await self.engine.engine_core.call_utility_async('prefill_lab_command',payload)
            if operation=='meta':result.update(token_manifest_sha256=digest(self.manifest),launch_argv=self.argv)
            return result

    async def init_state(self,engine_client,state,args):
        if engine_client is None:raise RuntimeError('An OpenAI-server EngineCore is required')
        if not hasattr(engine_client,'engine_core') or not hasattr(engine_client.engine_core,'call_utility_async'):
            raise RuntimeError('Unsupported vLLM engine utility channel')
        root=Path(__file__).resolve().parents[1]
        self.manifest=json.loads(Path(os.environ['PREFILL_LAB_MANIFEST']).read_text())
        cases=json.loads((root/'experiments/prefill-service-v1.cases.json').read_text())['cases']
        order=json.loads((root/'experiments/prefill-service-v1.order.json').read_text())['order']
        validate(self.manifest,cases,order)
        self.trials={t['trial_id']:t for t in order}
        transfers=json.loads((root/'experiments/prefill-service-v1.order.json').read_text())['transfer_slices']
        self.external_ids={t['trial_id'] for t in order if t['stage']=='http_bridge'}|{t['attempt_id'] for t in transfers}
        self.engine=engine_client
        import sys
        self.argv=sys.argv
