"""Private post-exec source-origin attestation; never imports a native hook."""

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time


def attestation(name, module, expected, event):
    path = Path(module.__file__).resolve()
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    raw_stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return dict(module=name, path=str(path), sha256=fingerprint,
                valid=str(path) == expected["path"] and fingerprint == expected["sha256"],
                event=event, pid=os.getpid(), start_ticks=int(raw_stat[19]),
                observed_at=time.monotonic())


def install():
    policy_path = os.environ.get("R24_CUT_CONSUMER_ORIGIN_POLICY")
    destination = os.environ.get("R24_CUT_CONSUMER_ORIGIN_RECORDS")
    if not policy_path or not destination:
        return
    policy = json.loads(Path(policy_path).read_bytes())
    directory = Path(destination)
    if (str(directory) != "/workspace/pr32-r23/cut-consumer-state01/origins-consumer01"
            or directory.is_symlink() or directory.stat().st_mode & 0o777 != 0o700):
        os._exit(73)
    expected = policy["origins"]

    def attest(name, module, event):
        record = attestation(name, module, expected[name], event)
        raw = (json.dumps(record, sort_keys=True) + "\n").encode()
        fd = os.open(directory / (str(os.getpid()) + ".jsonl"),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            if os.write(fd, raw) != len(raw):
                os._exit(73)
        finally:
            os.close(fd)
        if not record["valid"]:
            os._exit(73)

    class Loader:
        def __init__(self, base, name):
            self.base, self.name = base, name

        def create_module(self, spec):
            return self.base.create_module(spec) if hasattr(self.base, "create_module") else None

        def exec_module(self, module):
            self.base.exec_module(module)
            attest(self.name, module, "loaded_after_exec")

        def __getattr__(self, name):
            return getattr(self.base, name)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name not in expected:
                return None
            for finder in tuple(sys.meta_path):
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    if spec.loader is None or not hasattr(spec.loader, "exec_module"):
                        os._exit(73)
                    spec.loader = Loader(spec.loader, name)
                    return spec
            os._exit(73)

    def inherited():
        try:
            for name in expected:
                if name in sys.modules:
                    attest(name, sys.modules[name], "inherited_after_fork")
        except BaseException:
            os._exit(73)

    sys.meta_path.insert(0, Finder())
    os.register_at_fork(after_in_child=inherited)


try:
    install()
except BaseException:
    os._exit(73)
