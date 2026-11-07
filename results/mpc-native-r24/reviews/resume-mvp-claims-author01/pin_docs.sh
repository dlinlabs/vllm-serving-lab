#!/usr/bin/env bash
set -euo pipefail
set -C
R=/home/daniel_lin/gt/pr32/mayor/mpc-local-r21
DOCS="resume_project_summary.md architecture.md benchmark_results.md interview_defense.md resume_bullets.md"
mkdir pinned-docs
for f in $DOCS; do
  s="$R/docs/$f"
  test -f "$s"; test ! -L "$s"
  stat -c 'pre  %n size=%s mode=%a mtime=%y inode=%i links=%h' "$s"
  cp -- "$s" "pinned-docs/$f"
  chmod 444 "pinned-docs/$f"
done
( cd "$R/docs" && sha256sum -- $DOCS ) > DOC-PINS.sha256
chmod 444 DOC-PINS.sha256
cat DOC-PINS.sha256
( cd pinned-docs && sha256sum -c --strict ../DOC-PINS.sha256 )
for f in $DOCS; do stat -c 'post %n size=%s mode=%a mtime=%y inode=%i' "$R/docs/$f"; done
