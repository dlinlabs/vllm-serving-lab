# Slow-download setup timeout recovery

Dependency installation previously had a fixed 1800s subprocess wall-time
limit. A native rental installation reached that limit while downloading;
collection had not started. The download had been observed making progress.

Setup now accepts `--install-timeout SECONDS`, default 7200. It validates a
positive integer before running commands and reports that installation timeout
stops before CPU acceptance or collection. It preserves the existing venv and
pip cache: completed cached files may be reused; an interrupted file may need
to download again. There is no automatic retry, environment deletion, package
pin change, network-speed claim, or relaxation of experiment deadlines/gates.
The limit is a resource ceiling, not a runtime estimate.

**71 related local CPU/mock tests pass in 4.65s, with no unsuccessful cases.**
This selection covers default/custom setup limits, preservation of an existing
environment, timeout stopping before preparation, invalid limits, the existing
host and entry-HTTP regressions, and committed design/hash regeneration. No
new native install, full native CPU acceptance or GPU collection is claimed.
The prior 348-case HTTP-integration proof remains bound to ce01ad8 unchanged.

Recovery on the existing host: update the clean branch, then rerun setup with
`--next entry-http --install-timeout 7200` before starting `entry-http` only on
setup success. Keep the original failure log and use a separate resume log.
