"""Prepare a clean Linux/Python3.12 RTX4090 host for the PR32 calibration pilot.

Installs the pinned runtime, verifies CUDA/source files and runs native CPU
acceptance. Does not start collection. Use nohup to survive SSH disconnection.
"""
import json
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'


def main(*,operation='calibrate'):
    if operation not in ('calibrate','arrivals','clock'):raise ValueError('Invalid next experiment')
    if platform.system()!='Linux' or sys.version_info[:2]!=(3,12):
        raise RuntimeError('Use native Linux with Python3.12')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Use a clean tracked checkout')
    venv=ROOT/'.venv';python=venv/'bin/python'
    if not venv.exists():subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
    env=os.environ.copy();env['PATH']=str(venv/'bin')+os.pathsep+env.get('PATH',os.defpath)
    env['VIRTUAL_ENV']=str(venv);env.pop('PYTHONHOME',None)
    def run(*args,**kw):return subprocess.run([str(python),*args],cwd=ROOT,env=env,check=True,**kw)
    run('-c','import sys; assert sys.version_info[:2]==(3,12)')
    run('-m','pip','install','--only-binary=:all:','--extra-index-url','https://download.pytorch.org/whl/cu130',
        '-r','requirements-v1.txt','-r','requirements-cpu-tests.txt','ninja',timeout=1800)
    run('-m','pip','install','--no-deps','-e','.',timeout=300)
    run('-m','pip','check',timeout=60)
    run('-c', '''import json,torch
from prefill_lab.plugin import installed_sources
assert torch.__version__=='2.13.0+cu130' and torch.version.cuda=='13.0'
assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
x=torch.ones((32,32),device='cuda');y=x@x;torch.cuda.synchronize()
assert y[0,0].item()==32
print(json.dumps({'cuda_compute':'passed','verified_source_files':len(installed_sources()['files'])}))
''',timeout=120)
    directory=Path.home()/'pr32-inputs';directory.mkdir(exist_ok=True)
    pin=directory/'model.json';spec=dict(model=MODEL,revision=REVISION)
    if pin.exists() and json.loads(pin.read_text())!=spec:
        raise RuntimeError('Existing model pin differs; use a fresh host or explicitly preserve/move the old input directory')
    if not pin.exists():pin.write_text(json.dumps(spec)+'\n')
    run('scripts/prefill_host.py','prepare')
    print('PREPARED. Start the pilot: .venv/bin/python scripts/prefill_host.py '+operation,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--next',dest='operation',choices=('calibrate','arrivals','clock'),default='calibrate')
    main(operation=parser.parse_args().operation)
