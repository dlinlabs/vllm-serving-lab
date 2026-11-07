#!/usr/bin/env bash
set -u
set -C
log="$1"; shift
case "$log" in */*|""|commands.log) echo "bad log name" >&2; exit 64;; esac
{ printf '# %s cwd=%s\n' "$(date -u +%FT%TZ)" "$PWD"; printf '%q ' "$@"; printf '> %q 2>&1\n' "$log"; } >> commands.log
"$@" > "$log" 2>&1
rc=$?
printf 'exit=%s\n' "$rc" >> commands.log
exit "$rc"
