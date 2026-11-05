#!/usr/bin/env bash
set -euo pipefail
set -C
R=/home/daniel_lin/gt/pr32/mayor/mpc-local-r21
C="$PWD"
n=$(grep -c . SELECTED-REFS.txt); test "$n" -le 10
total=0
while IFS= read -r p; do
  case "$p" in /*|*..*|"") echo "bad path $p"; exit 3;; esac
  f="$R/$p"; test -f "$f"; test ! -L "$f"
  rp=$(realpath -e "$f"); case "$rp" in "$R"/*) ;; *) echo "outside root $rp"; exit 4;; esac
  s=$(stat -c %s "$f"); total=$((total+s))
  stat -c 'pre %n size=%s mode=%a mtime=%y' "$f"
done < SELECTED-REFS.txt
echo "count=$n total_bytes=$total limit=1048576"; test "$total" -le 1048576
( cd "$R" && xargs -d '\n' sha256sum -- < "$C/SELECTED-REFS.txt" ) > REF-PINS.sha256
chmod 444 REF-PINS.sha256
cat REF-PINS.sha256
( cd "$R" && sha256sum -c --strict "$C/REF-PINS.sha256" )
