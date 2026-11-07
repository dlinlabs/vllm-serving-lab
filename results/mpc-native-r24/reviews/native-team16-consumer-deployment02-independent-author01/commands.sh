#!/bin/sh
# Exact commands for d2r01, in execution order, run from this directory. Source reads (view/awk/grep of pinned
# subject files) are not listed. No controls, Native, GPU, SSH, network or git. Every write is exclusive (set -C or open "x").
P="env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 nice -n 19"
PY314=/home/daniel_lin/gt/pr32/mayor/mpc-local-r21/.venv/bin/python3.14
PY312=/home/daniel_lin/.copilot/session-state/73eaad21-e7ed-4c46-9f3f-a2626aa1cc3d/files/r24-mpc-py312-venv/bin/python3.12
exit 0

set -C
date -u +%Y-%m-%dT%H:%M:%S.%NZ > MARKER-review-start.txt
# PLAN.json written (questions, budget, first selection), then:
mkdir verification
$P $PY314 -B pin_d2r01.py > verification/pin-py314.log; echo "exit $?" > verification/pin-py314.exit
date -u +%Y-%m-%dT%H:%M:%S.%NZ > MARKER-freeze.txt
$P $PY314 -B add_pins_d2r01.py > verification/add-pins-py314.log; echo $? > verification/add-pins-py314.exit
$P $PY314 -B add_pins_d2r01_02.py > verification/add-pins-02-py314.log; echo $? > verification/add-pins-02-py314.exit
$P $PY314 -B make_report_d2r01.py > verification/make-report-py314.log; echo $? > verification/make-report-py314.exit
{ echo "check1: subject and related sealed roots newer than MARKER-review-start.txt (expected none)"
  find ../r24-team16-native-cut-consumer-deployment02 ../r24-team16-native-cut-consumer-successor01 \
       ../r24-team16-native-cut-consumer-successor-independent-review01 \
       ../r24-team16-native-cut-consumer-successor-independent-review01-disposition01 \
       -newer MARKER-review-start.txt; echo "find_exit=$?"; } > verification/no-writes-after-marker.txt
$P $PY314 -B verify_d2r01.py --pre-seal > verification/verify-pre-seal-py314.json; echo $? > verification/verify-pre-seal-py314.exit
$P $PY312 -B verify_d2r01.py --pre-seal > verification/verify-pre-seal-py312.json; echo $? > verification/verify-pre-seal-py312.exit
find . -type f ! -path ./SHA256SUMS | sed 's|^\./||' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS; chmod a-w SHA256SUMS
$P $PY314 -B verify_d2r01.py
$P $PY312 -B verify_d2r01.py
