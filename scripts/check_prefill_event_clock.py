"""Recompute native clock confirmation with the exact captured checkout."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_prefill_arrivals import audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    report=audit(a.directory,event_clock=True)
    print(json.dumps(dict(audit_valid=True,candidate_passed=report['candidate_passed'],model_accepted=False)))
    return 0 if report['candidate_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
