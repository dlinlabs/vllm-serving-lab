"""Capture process output without ever persisting the per-attempt admin credential."""

import json
import os
from pathlib import Path
import subprocess
import sys


def redacted_chunks(chunks, secret):
    if not secret:
        raise ValueError("nonempty_private_redaction_token_required")
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while (at := buffer.find(secret)) >= 0:
            yield buffer[:at] + b"[REDACTED_AUTH]"
            buffer = buffer[at + len(secret):]
        count = max(0, len(buffer) - len(secret) + 1)
        if count:
            yield buffer[:count]
            buffer = buffer[count:]
    yield buffer.replace(secret, b"[REDACTED_AUTH]")


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise ValueError("private_config_and_exact_child_command_required")
    config_path = Path(sys.argv[1])
    if config_path.is_symlink() or config_path.stat().st_mode & 0o077:
        raise ValueError("private_config_permissions_required")
    key = json.loads(config_path.read_bytes())["ADMIN_API_KEY"].encode()
    process = subprocess.Popen(sys.argv[3:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for chunk in redacted_chunks(iter(lambda: process.stdout.read1(8192), b""), key):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return process.wait()
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
