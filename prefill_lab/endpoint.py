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
            if not isinstance(payload,dict) or set(payload)-{'operation','trial_id','reason','observer','frozen','runtime'}:
                raise HTTPException(400,'Unexpected command fields')
            operation=payload.get('operation')
            if operation not in ('meta','start','status','abort','finish','external_begin','external_end','live_configure'):
                raise HTTPException(400,'Unknown experiment operation')
            if operation=='live_configure' and not self.live_enabled:raise HTTPException(400,'Live bridge is disabled')
            if operation!='live_configure' and set(payload)&{'frozen','runtime'}:raise HTTPException(400,'Unexpected live fields')
            if operation=='start':
                tid=payload.get('trial_id')
                if tid not in self.trials:raise HTTPException(400,'Trial is not frozen')
                trial=dict(self.trials[tid])
                trial['requests']=[dict(r,input_ids=self.manifest['pools'][r['token_key']]['input_ids'])
                                   for r in self.manifest['trials'][tid]]
                payload=dict(operation='start',trial=trial)
            if operation=='external_begin':
                live_trial=payload.get('trial_id') in getattr(self,'live_trials',{})
                allowed=('off','light') if live_trial else ('light',)
                if payload.get('trial_id') not in self.external_ids or payload.get('observer','light') not in allowed:
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
        calibration=os.environ.get('PREFILL_LAB_EFFECTIVE_CALIBRATION','0')
        arrivals=os.environ.get('PREFILL_LAB_DELAYED_ARRIVALS','0')
        clock=os.environ.get('PREFILL_LAB_EVENT_CLOCK','0')
        if clock not in ('0','1') or (clock=='1' and (arrivals!='1' or calibration!='0')):
            raise ValueError('Invalid/conflicting event-clock mode')
        if arrivals not in ('0','1') or (arrivals=='1' and calibration!='0'):raise ValueError('Invalid/conflicting delayed mode')
        if calibration not in ('0','1'):raise ValueError('Invalid calibration mode')
        probe=os.environ.get('PREFILL_LAB_QUEUE_PROBE','0')
        if probe not in ('0','1') or (probe=='1' and any(x!='0' for x in (clock,arrivals,calibration,os.environ.get('PREFILL_LAB_LIVE_STATE','0')))):
            raise ValueError('Invalid/conflicting queue probe mode')
        confirmation=os.environ.get('PREFILL_LAB_QUEUE_CONFIRMATION','0')
        if confirmation not in ('0','1') or (confirmation=='1' and any(x!='0' for x in (probe,clock,arrivals,calibration,os.environ.get('PREFILL_LAB_LIVE_STATE','0')))):
            raise ValueError('Invalid/conflicting queue confirmation mode')
        if confirmation=='1':
            from .queue_confirmation import protocol,validate_manifest
            plan=protocol();cases,order=plan['cases'],plan['order'];validate_manifest(self.manifest)
        elif probe=='1':
            from .queue_probe import protocol,validate_manifest
            plan=protocol();cases,order=plan['cases'],plan['order'];validate_manifest(self.manifest)
        elif clock=='1':
            from .event_clock_calibration import protocol,validate_manifest
            plan=protocol();cases,order=plan['cases'],plan['order'];validate_manifest(self.manifest)
        elif arrivals=='1':
            from .delayed_calibration import protocol,validate_manifest
            plan=protocol();cases,order=plan['cases'],plan['order'];validate_manifest(self.manifest)
        elif calibration=='1':
            from .effective_calibration import protocol,validate_manifest
            plan=protocol();cases,order=plan['cases'],plan['order']
            validate_manifest(self.manifest)
        validate(self.manifest,cases,order)
        self.trials={t['trial_id']:t for t in order}
        transfers=json.loads((root/'experiments/prefill-service-v1.order.json').read_text())['transfer_slices']
        self.external_ids={t['trial_id'] for t in order if t['stage']=='http_bridge'}|{t['attempt_id'] for t in transfers}
        if calibration=='1' or arrivals=='1' or probe=='1' or confirmation=='1':self.external_ids=set()
        live_flag=os.environ.get('PREFILL_LAB_LIVE_STATE','0')
        if live_flag not in ('0','1'):raise ValueError('Invalid live-state launch flag')
        self.live_enabled=live_flag=='1';self.live_trials={}
        if self.live_enabled:
            if clock!='1':raise ValueError('Live bridge requires the dedicated clock launch')
            from .live_http import protocol
            self.live_trials={t['trial_id']:t for t in protocol()['order']}
            self.external_ids=set(self.live_trials)
        self.engine=engine_client
        import sys
        self.argv=sys.argv
