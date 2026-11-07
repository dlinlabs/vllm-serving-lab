"""Freeze a committed, accepted source package locally; never contact a GPU host."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file()
            and (parent / "experiments/mpc-research/resume-mvp/nominal_protocol.json").is_file())
INVENTORY = "results/mpc-native-r24/reviews/resume-p0-native-package-candidate01-proof/source-inventory.json"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-reference", type=Path, required=True)
    args = parser.parse_args()
    commit = git("rev-parse", args.source_commit + "^{commit}").decode().strip()
    assert len(commit) == 40
    output = args.output.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    rows = json.loads(git("show", commit + ":" + INVENTORY))["files"]
    assert len(rows) == 62 and len({row["path"] for row in rows}) == 62
    manifest = []
    for row in sorted(rows, key=lambda item: item["path"]):
        relative = Path(row["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        data = git("show", commit + ":" + row["path"])
        mode = git("ls-tree", commit, "--", row["path"]).split()[0]
        assert mode in (b"100644", b"100755")
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if mode == b"100755" else 0o644)
        manifest.append({"path": row["path"], "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})
    with tarfile.open(output / "source.tar", "w") as archive:
        for row in manifest:
            info = tarfile.TarInfo(row["path"])
            info.size = row["bytes"]
            info.mode = (source / row["path"]).stat().st_mode & 0o777
            info.mtime = info.uid = info.gid = 0
            archive.addfile(info, io.BytesIO((source / row["path"]).read_bytes()))
    value = {"source_commit": commit, "files": manifest, "file_count": len(manifest),
             "source_tar_sha256": hashlib.sha256((output / "source.tar").read_bytes()).hexdigest(),
             "source_tar_bytes": (output / "source.tar").stat().st_size,
             "scope": "CPU-frozen package, no GPU execution authority"}
    (output / "source-manifest.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    check = r'''
import json
from pathlib import Path
import sys
from dataclasses import asdict
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root), str(root / "experiments/mpc-research/resume-mvp")]
import native_nominal_qualification
import nominal_client
import nominal_episode
import nominal_runtime
from mpc_lab.contracts import CostSpec, ModelParameters
expected = json.loads(Path(sys.argv[2]).read_text())["source"]
parameters = ModelParameters(cost_spec=CostSpec(**expected["cost_spec"]))
assert asdict(parameters.identity) == expected["identity"]
for module in tuple(sys.modules.values()):
    name = getattr(module, "__name__", "")
    if name.startswith(("mpc_lab", "prefill_lab", "nominal_")) or name in (
        "native_nominal_qualification", "native_services_smoke", "telemetry", "trace_workload"):
        assert Path(module.__file__).resolve().is_relative_to(root), name
factory, binding = nominal_runtime.load_train_scenario(nominal_episode.TRAIN, purpose="engineering")
assert callable(factory) and binding["purpose"] == "engineering"
print(json.dumps({
    "scope": "ISOLATED_CPU_IMPORT_AND_TRAIN_BINDING_ONLY",
    "modules_from_package": True,
    "historical_identity_fixture_reconstructed": True,
    "reference_is_current_live_authority": False,
    "prior_content_sha256": binding["prior_content_sha256"],
    "module_paths": nominal_runtime.module_paths(),
    "Native_qualified": False,
}, sort_keys=True))
'''
    (output / "import_check.py").write_text(check)
    command = [sys.executable, "-I", "-B", str(output / "import_check.py"),
               str(source), str(args.historical_reference.resolve())]
    (output / "import-command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, cwd=output, capture_output=True, text=True)
    (output / "import.stdout.json").write_text(result.stdout)
    (output / "import.stderr.log").write_text(result.stderr)
    (output / "import.exit-code").write_text(str(result.returncode) + "\n")
    if result.returncode:
        raise RuntimeError("isolated package import/binding failed; evidence retained")
    print(json.dumps({"source_commit": commit, "package_files": len(manifest),
                      "package_sha256": value["source_tar_sha256"],
                      "isolated_import": "PASS", "GPU_contacted": False}))


if __name__ == "__main__":
    main()
