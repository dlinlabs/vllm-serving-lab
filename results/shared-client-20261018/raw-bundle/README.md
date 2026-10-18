# Complete original uploads

The three archives are stored as ordered binary parts, each at most 1 MiB.
`index.json` records every part, reconstructed archive and original regular
file's byte size and SHA256. The archive bytes are unchanged from the uploads.
Do not concatenate parts from different archive IDs.

From the repository root:

```bash
python3 results/shared-client-20261018/extract_bundle.py --output-dir /tmp/shared-client-captures
```

The output has separate `failed-a/`, `shared-a/` and `shared-b/` trees, preserving
each archive's original layout without overwriting another capture's evidence.
Use a fresh output path. The helper verifies all selected parts, archive bytes,
member paths, sizes and per-file hashes before publishing the extracted tree.
It rejects links, traversal paths, undeclared/duplicate/missing files and an
existing output directory. `--archives shared-a shared-b` can select a subset.

- `failed-a`: original per-request excitation A stopped at the 210 s control
  deadline; includes its smoke, native CPU and setup evidence. It stays invalid.
- `shared-a`: qualified smoke/A, frozen qualification file, native CPU and setup.
- `shared-b`: qualified fresh-host smoke/B, external frozen B file, native CPU,
  setup and outer logs.

The `captures/`, `setup/` and `native-validation/` siblings are browsing subsets.
They do not replace these complete raw inputs. No held-out C capture is included.
