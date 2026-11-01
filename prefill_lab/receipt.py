"""Opt-in CPU receipt evidence; never an input to a pre-arrival forecast."""
from .contracts import Buffer, finite


class ReceiptRecorder:
    def __init__(self, clock, capacity=8192):
        self.clock, self.events = clock, Buffer(capacity)
        self.pending = {}; self.scheduled = self.committed = 0

    def append(self, kind, **values):
        row = dict(kind=kind, at=self.clock(), **values)
        self.events.append(row); return row

    def schedule(self, output):
        if not output.num_scheduled_tokens:return
        key=id(output)
        if key in self.pending:raise ValueError('Duplicate outstanding schedule')
        self.scheduled+=1; self.pending[key]=self.scheduled
        self.append('schedule',sequence=self.scheduled)

    def commit(self, output):
        if not output.num_scheduled_tokens:return
        sequence=self.pending.pop(id(output)); self.committed+=1
        self.append('commit',sequence=sequence)

    def release(self):self.append('release')

    def receive(self,rid,posted,enqueue,scheduler):
        row=self.append('receipt',request_id=rid,posted=posted,enqueue=enqueue,
            scheduled=self.scheduled,committed=self.committed,pending_sequences=sorted(self.pending.values()))
        short=lambda key:key.rsplit('/',1)[-1]
        row['state']=dict(running=[short(r.request_id) for r in scheduler.running],
            waiting=[short(r.request_id) for r in scheduler.waiting],
            requests={short(key):dict(computed=int(r.num_computed_tokens),tokens=int(r.num_tokens),
                placeholders=int(r.num_output_placeholders)) for key,r in scheduler.requests.items()})
        row['observation_end']=self.clock()

    def capture(self):
        return dict(schema='receipt-events-v1',events=self.events.rows,overflow=self.events.overflow,pending=len(self.pending))


def validate(record,capture):
    if record.get('schema')!='receipt-events-v1' or record['overflow'] or record['pending']:
        raise ValueError('Incomplete receipt evidence')
    events=record['events']; pending=set();scheduled=committed=releases=0;receipts={}
    release_counts=None;last_schedule=last_commit=None
    if not events or any(not finite(e.get('at')) for e in events):raise ValueError('Invalid receipt clock')
    if [e['at'] for e in events]!=sorted(e['at'] for e in events):raise ValueError('Receipt chronology')
    for e in events:
        kind=e['kind']
        if kind=='schedule':
            scheduled+=1
            if type(e['sequence']) is not int or e['sequence']!=scheduled:raise ValueError('Schedule sequence')
            pending.add(scheduled);last_schedule=e['at']
        elif kind=='commit':
            if type(e['sequence']) is not int or e['sequence'] not in pending or e['sequence']!=committed+1:
                raise ValueError('Unmatched/reordered commit')
            pending.remove(e['sequence']);committed+=1;last_commit=e['at']
        elif kind=='release':
            releases+=1;release_counts=(scheduled,committed)
            if not 0<=e['at']-capture['release']<=.005:raise ValueError('Release anchor differs')
        elif kind=='receipt':
            rid=e['request_id']
            if releases!=1 or rid in receipts or rid not in capture['requests']:raise ValueError('Receipt identity')
            r=capture['requests'][rid]
            if (e['scheduled'],e['committed'],e['pending_sequences'])!=(scheduled,committed,sorted(pending)):
                raise ValueError('Receipt counters differ')
            times=[e['posted'],e['enqueue'],e['at'],e['observation_end'],r['first']]
            if not all(finite(v) for v in times) or times!=sorted(times):raise ValueError('Receipt/request chronology')
            if e['posted']!=r['queue_posted'] or e['enqueue']!=r['enqueue']:raise ValueError('Request timestamps differ')
            state=e['state'];requests=state['requests'];names={k.rsplit('/',1)[-1] for k in capture['requests']}
            if not set(requests)<=names or rid.rsplit('/',1)[-1] in requests:raise ValueError('Invalid pre-ADD state')
            members=state['running']+state['waiting']
            if len(set(members))!=len(members) or not set(members)<=set(requests):raise ValueError('Scheduler membership')
            if any(set(v)!={'computed','tokens','placeholders'} or any(type(x) is not int or x<0 for x in v.values()) for v in requests.values()):
                raise ValueError('Invalid request state')
            receipts[rid]=dict(e,schedules_since_release=scheduled-release_counts[0],
                commits_since_release=committed-release_counts[1],last_schedule=last_schedule,last_commit=last_commit)
        else:raise ValueError('Unknown receipt event')
    wanted={rid for rid,r in capture['requests'].items() if r['role']=='sentinel'}
    if pending or releases!=1 or set(receipts)!=wanted:raise ValueError('Missing receipt/release or pending events')
    if any(e['at']<capture['start'] or e['at']>capture['end'] for e in events):raise ValueError('Event outside capture')
    if capture.get('observer')=='light':
        starts=[e for e in events if e['kind']=='schedule'];ends=[e for e in events if e['kind']=='commit']
        if len(starts)!=len(capture['steps']) or len(ends)!=len(starts):raise ValueError('LIGHT step count differs')
        if any(not s['start']<=a['at']<=b['at']<=s['end'] for a,b,s in zip(starts,ends,capture['steps'])):
            raise ValueError('LIGHT chronology differs')
    return receipts
