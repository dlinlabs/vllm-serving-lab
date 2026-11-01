"""CPU-only busy snapshot forecast; provided metadata is not native qualification."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prefill_lab import event_clock_calibration as clock
from prefill_lab.async_predictor import Request
from prefill_lab.contracts import save
from prefill_lab.live_predictor import forecast
from scripts.predict_prefill_async import runtime_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','request','runtime','freeze','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();load=lambda path:json.loads(path.read_text());runtime=load(a.runtime);frozen=load(a.freeze)
    clock.predict_frozen(frozen,clock.protocol(),runtime)
    scope,dispatch=runtime_inputs(runtime);model=clock.costs(frozen['parameters_s'],clock.protocol(),clock.specification())
    result=forecast(load(a.snapshot),Request(**load(a.request)),model,dispatch,scope=scope)
    a.output.mkdir(parents=True,exist_ok=False);save(a.output/'forecast.json',result)
    print(json.dumps(dict(ttft_from_cut_s=result['ttft_from_cut_s'],model_accepted=False,http_accepted=False)))


if __name__=='__main__':main()
