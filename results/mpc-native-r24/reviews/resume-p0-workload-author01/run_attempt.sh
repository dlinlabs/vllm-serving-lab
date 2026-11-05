#!/usr/bin/env bash
# usage (from the capsule directory): bash run_attempt.sh NAME CMD [ARG...]
# Writes NAME.start, NAME.log (stdout+stderr), NAME.exit and NAME.end, all with noclobber: an
# attempt name can never be reused, so every failed attempt is preserved and a retry needs a new
# attempt number. Never reruns, retries or edits anything.
set -u
set -C
if [ "$#" -lt 2 ]; then
    echo "usage: bash run_attempt.sh NAME CMD [ARG...]" >&2
    exit 96
fi
name=$1
shift
date -u +%Y-%m-%dT%H:%M:%SZ > "$name.start" || exit 97
"$@" > "$name.log" 2>&1
status=$?
printf '%s\n' "$status" > "$name.exit" || exit 98
date -u +%Y-%m-%dT%H:%M:%SZ > "$name.end" || exit 99
exit "$status"
