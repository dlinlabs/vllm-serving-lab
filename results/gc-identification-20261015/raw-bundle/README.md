# Complete original capture

The numbered parts concatenate to the **byte-identical uploaded gzip archive**.
Nothing was removed, redacted, regenerated or substituted. The manifest records
archive/part sizes and SHA-256 values plus all 78 original file hashes.

Do not try to extract an individual part. From the repository root:

```bash
python3 results/gc-identification-20261015/extract_bundle.py --output-dir /tmp/gc-identification-20261015
```

The helper verifies the archive, refuses a nonempty destination, checks safe member
paths and validates every restored file. No GPU or Python package installation is
needed for extraction. See [the findings](../report.md) for offline audit commands.
