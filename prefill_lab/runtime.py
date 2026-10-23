"""Engine-thread trial state; arrival timers only post to the existing input queue."""
import threading
import time
from .contracts import Buffer, features, fifo_reference, scheduled_work, step_key

INTERNAL_REQUEST = 'prefill-lab-internal-arrival-v1'


def snapshot(scheduler):
    return dict(running=[r.request_id for r in scheduler.running],
                waiting=[r.request_id for r in scheduler.waiting],
                requests={rid:dict(computed=int(r.num_computed_tokens),
                                   num_tokens=int(r.num_tokens),
                                   placeholders=int(r.num_output_placeholders),
                                   prompt_tokens=int(r.num_prompt_tokens),max_tokens=int(r.max_tokens))
                          for rid,r in scheduler.requests.items()})


class Trial:
    """Single trial, no retry. Mutations run on the EngineCore thread only."""
    def __init__(self, core, specification, request_factory, *, clock=time.monotonic, capacity=8192):
        self.core, self.spec, self.factory, self.clock = core, specification, request_factory, clock
        self.id, self.mode = specification['trial_id'], specification['observer']
        self.started, self.release, self.done = clock(), None, None
        self.errors, self.steps, self.requests = [], Buffer(capacity), {}
        self.pending, self.enqueued, self.timers = {}, set(), []
        self.cancel = threading.Event()
        self.inflight, self.status = {}, 'running'
        for row in specification['requests']:
            rid = f'plab/{self.id}/{row["id"]}'
            if rid in self.requests:raise ValueError('Duplicate request ID')
            self.pending[rid] = dict(row,id=rid)
            self.requests[rid] = dict(role=row['role'], input_tokens=len(row['input_ids']),
                                      max_tokens=row['max_tokens'], output_tokens=0,
                                      first=None,finished=None,enqueue=None,queue_posted=None)
        self.backgrounds = {k for k,r in self.requests.items() if r['role']=='background'}
        self.foreground = set(self.requests)-self.backgrounds
        self.release_state = None
        self.epoch = specification['epoch']

    def begin(self):
        if self.backgrounds:
            self.inject(list(self.backgrounds),self.clock())
        else:
            self.release_foreground()

    def inject(self, identifiers, posted):
        # Called from the engine thread; the entire batch is inserted before scheduling.
        now = self.clock()
        for rid in sorted(identifiers):
            if rid in self.enqueued:raise ValueError('Duplicate engine enqueue')
            row = self.pending[rid]
            self.core.add_request(self.factory(row))
            self.enqueued.add(rid)
            self.requests[rid].update(enqueue=now,queue_posted=posted,
                                      target=None if self.release is None else self.release+row['at'])
            if self.release is not None and row['at'] and abs(posted-(self.release+row['at'])) > .005:
                self.errors.append('Sentinel input-queue injection lag exceeds5ms')
        # Queue residence/admission delay is reported separately, never hidden as timer lag.

    def release_foreground(self):
        if self.release is not None:raise RuntimeError('Duplicate foreground barrier')
        self.release = self.clock()
        self.release_state = snapshot(self.core.scheduler)
        immediate = [rid for rid in self.foreground if self.pending[rid]['at']==0]
        self.inject(immediate,self.release)
        for rid in sorted(self.foreground-set(immediate)):
            target = self.release+self.pending[rid]['at']
            def arrival(rid=rid,target=target):
                if self.cancel.wait(max(0,target-self.clock())):return
                self.core.input_queue.put_nowait((INTERNAL_REQUEST,(self.epoch,[rid],self.clock())))
            timer = threading.Thread(target=arrival,daemon=True,name='prefill-arrival')
            self.timers.append(timer);timer.start()

    def before_schedule(self):
        return self.clock(), snapshot(self.core.scheduler)

    def after_schedule(self, output, before):
        if not output.num_scheduled_tokens:return
        start,state = before
        work = scheduled_work(output)
        for r in work:r['prompt_length']=self.requests[r['request_id']]['input_tokens']
        if any(r['request_id'] not in self.enqueued for r in work):
            self.errors.append('Unexpected external request in instrumented scheduler')
            raise RuntimeError(self.errors[-1])
        if getattr(output,'preempted_req_ids',None) or getattr(output,'scheduled_spec_decode_tokens',None):
            self.errors.append('Preemption/speculation violates matched-work protocol')
        key = step_key(output)
        if key in self.inflight:raise RuntimeError('Duplicate in-flight step ID')
        counts = dict(output.num_scheduled_tokens)
        expected = fifo_reference(state,self.core.vllm_config.scheduler_config.max_num_batched_tokens,
                                  self.core.vllm_config.scheduler_config.max_num_seqs,
                                  self.core.vllm_config.scheduler_config.long_prefill_token_threshold)
        row = dict(step_id=key,start=start,end=None,work=work,state_before=state,
                   phase='foreground' if self.release is not None and self.done is None else 'background',
                   parity=dict(matches=counts==expected,expected=expected,actual=counts),
                   **features(work,{rid:r['input_tokens'] for rid,r in self.requests.items()}))
        if self.mode!='off':
            try:
                self.steps.append(row)
                self.inflight[key] = row
            except RuntimeError:
                self.errors.append('Capture buffer overflow; observations incomplete')
        if self.release is not None and self.done is None:
            if any(self.requests[rid]['finished'] is not None for rid in self.backgrounds):
                self.errors.append('D16 backgrounds did not outlive foreground')

    def after_output(self, output, results):
        now = self.clock()
        if self.mode!='off' and output.num_scheduled_tokens:
            key=step_key(output)
            if key not in self.inflight:self.errors.append('Output missing scheduled step ID')
            else:self.inflight.pop(key)['end']=now
        for batch in results.values():
            retained=[]
            for item in batch.outputs:
                rid=item.request_id
                if rid not in self.requests:
                    retained.append(item);continue
                r=self.requests[rid]
                count=len(item.new_token_ids)
                if count and r['first'] is None:r['first']=now
                r['output_tokens']+=count
                if item.finish_reason is not None:
                    r['finished']=now;r['finish_reason']=str(item.finish_reason)
            # Private trial requests were never registered with frontend OutputProcessor.
            batch.outputs=retained
        if self.release is None and all(self.requests[k]['first'] is not None for k in self.backgrounds):
            self.release_foreground()
        if self.done is None and all(self.requests[k]['finished'] is not None for k in self.foreground):
            self.done=now
            if any(self.requests[k]['finished'] is not None and self.requests[k]['finished']<now for k in self.backgrounds):
                self.errors.append('Background finished before foreground')
        if len(self.enqueued)==len(self.requests) and all(r['finished'] is not None for r in self.requests.values()):
            self.status='complete';self.cancel.set()

    def capture(self):
        return dict(schema='prefill-capture-v1',trial_id=self.id,observer=self.mode,status=self.status,
                    start=self.started,release=self.release,foreground_done=self.done,
                    end=self.clock(),release_state=self.release_state,requests=self.requests,
                    steps=self.steps.rows,overflow=self.steps.overflow,errors=self.errors,
                    pending_step_ids=sorted(self.inflight))

    def abort(self,reason):
        self.cancel.set();self.errors.append(reason);self.status='invalid'
        self.core.abort_requests(list(self.enqueued))


class ExternalCapture(Trial):
    """Observe normal HTTP/gateway traffic without intercepting frontend outputs."""
    def __init__(self,core,trial_id,mode,expected_foreground=None):
        self.core,self.id,self.mode,self.clock=core,trial_id,mode,time.monotonic
        self.started,self.release,self.done=self.clock(),None,None
        self.requests,self.inflight,self.enqueued={}, {}, set()
        self.errors,self.steps=[],Buffer()
        self.backgrounds,self.foreground=set(),set()
        self.release_state=None
        self.status='running';self.cancel=threading.Event();self.expected_foreground=expected_foreground

    def register_request(self,request):
        rid=request.request_id
        if rid in self.requests:raise ValueError('Duplicate external request ID')
        metadata=request.sampling_params.extra_args or {}
        role=metadata.get('prefill_lab_role','transfer')
        if role not in ('background','foreground','sentinel','transfer'):
            raise ValueError('Invalid external role')
        self.requests[rid]=dict(role=role,input_tokens=request.num_prompt_tokens,
                               max_tokens=request.max_tokens,output_tokens=0,first=None,
                               finished=None,enqueue=self.clock(),queue_posted=None)
        self.enqueued.add(rid)
        if role=='background':self.backgrounds.add(rid)
        else:
            self.foreground.add(rid)
            if self.release is None:
                self.release=self.clock();self.release_state=snapshot(self.core.scheduler)

    def after_output(self,output,results):
        now=self.clock()
        if self.mode!='off' and output.num_scheduled_tokens:
            key=step_key(output)
            if key not in self.inflight:self.errors.append('Unmatched external output step')
            else:self.inflight.pop(key)['end']=now
        for batch in results.values():
            for item in batch.outputs:
                r=self.requests.get(item.request_id)
                if r is None:
                    self.errors.append('Unknown external output');continue
                count=len(item.new_token_ids)
                if count and r['first'] is None:r['first']=now
                r['output_tokens']+=count
                if item.finish_reason is not None:
                    r['finished']=now;r['finish_reason']=str(item.finish_reason)

        if self.expected_foreground is not None and len(self.foreground)==self.expected_foreground and all(self.requests[r]['finished'] is not None for r in self.foreground):
            if self.done is None:self.done=now
