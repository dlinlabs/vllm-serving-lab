#!/bin/bash
# usage: run_attempt.sh <attempt-name> <absolute-python>; ONE run, no retries; failures preserved; new dir only
set -u
ROOT=/home/daniel_lin/gt/pr32/mayor/mpc-local-r21
S=$ROOT/.runtime/r24-resume-p0-nominal-metrics02
SRC=$ROOT/experiments/mpc-research/resume-mvp
A=$S/attempts/$1
PY=$2
PINNED=(experiments/mpc-research/resume-mvp/nominal_metrics.py experiments/mpc-research/resume-mvp/nominal_client.py
        experiments/mpc-research/resume-mvp/test_nominal_metrics.py experiments/mpc-research/resume-mvp/test_nominal_client.py)
DEPS=(trace_workload.py telemetry.py experiments/mpc-research/native-r23/score_http.py)
set -C
mkdir -p -m 700 "$S/attempts"
mkdir -m 700 "$A" || { echo "attempt dir exists: refused" >&2; exit 9; }
mkdir -m 700 "$A/pinned"
(cd "$ROOT" && sha256sum "${PINNED[@]}" "${DEPS[@]}") > "$A/pins-before.sha256"
for f in "${PINNED[@]}"; do cp -p "$ROOT/$f" "$A/pinned/"; done
(cd "$SRC" && ls -la --time-style=full-iso __pycache__) > "$A/pycache-before.txt" 2>&1
cat > "$A/run.sh" <<RUN
#!/bin/bash
# $1: single run, no retries; failures preserved
cd $SRC
exec env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 120 nice -n19 $PY -B -m pytest -q -p no:cacheprovider --noconftest -c $S/pytest-empty.ini --rootdir=$A --basetemp=$A/basetemp test_nominal_metrics.py test_nominal_client.py
RUN
chmod 555 "$A/run.sh"
date -u +%FT%T.%3NZ > "$A/started.txt"
bash "$A/run.sh" > "$A/log.txt" 2>&1
echo $? > "$A/exit_code.txt"
date -u +%FT%T.%3NZ > "$A/ended.txt"
(cd "$ROOT" && sha256sum "${PINNED[@]}" "${DEPS[@]}") > "$A/pins-after.sha256"
(cd "$SRC" && ls -la --time-style=full-iso __pycache__) > "$A/pycache-after.txt" 2>&1
{ cmp "$A/pins-before.sha256" "$A/pins-after.sha256" && cmp "$A/pycache-before.txt" "$A/pycache-after.txt"; echo "unchanged_exit=$?"; } > "$A/unchanged.txt" 2>&1
chmod -R a-w "$A"
echo "$1 exit=$(cat "$A/exit_code.txt") $(tail -n 1 "$A/log.txt")"
