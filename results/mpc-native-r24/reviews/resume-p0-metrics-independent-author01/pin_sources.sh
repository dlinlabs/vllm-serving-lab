#!/usr/bin/env bash
# Pin selected sources: metadata, sha256 vs expected, byte copy (444). Exit 3 on any mismatch.
set -u
R=/home/daniel_lin/gt/pr32/mayor/mpc-local-r21
mkdir -p pinned-src
bad=0
while read -r rel exp; do
  [ -z "$rel" ] && continue
  p="$R/$rel"
  if [ ! -f "$p" ] || [ -L "$p" ]; then echo "MISSING_OR_SYMLINK $rel"; bad=1; continue; fi
  stat -c 'STAT %n size=%s mode=%a mtime=%y' "$p"
  act=$(sha256sum "$p" | cut -d' ' -f1)
  case "$act" in "$exp"*) echo "MATCH $act $rel";; *) echo "MISMATCH expected=$exp actual=$act $rel"; bad=1;; esac
  b=$(basename "$rel")
  cp -p -n "$p" "pinned-src/$b" && chmod 444 "pinned-src/$b"
  cpy=$(sha256sum "pinned-src/$b" | cut -d' ' -f1)
  [ "$cpy" = "$act" ] && echo "COPY_OK $b" || { echo "COPY_DIFF $b"; bad=1; }
  echo "$act  $rel" >> SOURCE-PINS.sha256
done < SELECTED-SOURCES.txt
[ "$bad" = 0 ] && echo ALL_PINNED || { echo PIN_FAILURE; exit 3; }
