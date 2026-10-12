# Complete diagnostics archive

These 11 binary parts concatenate to the **unmodified** uploaded
`vllm-diag-complete.tar.gz` (21,123,273 bytes; 81 original files).
[manifest.json](manifest.json) records full-archive and per-part hashes.
Parts avoid the connector transport size limit; they do not change the raw data.

From the repository root, verify every part and extract without a GPU:

```bash
bundle=$(mktemp -d)
python results/diagnostics-20261012/extract_bundle.py --output-dir "$bundle"
```

Use the extracted directory with the reproduction instructions in
[the analysis report](../report.md). All raw traces, gateway/engine events, setup
logs, CPU test and external-only validation evidence, and the original analysis
are included. The smaller report/manifest copies elsewhere in the repository
are provided only for convenient browsing.
