"""Capture actual source bytes as well as Git state for reproducibility."""
from pathlib import Path
import hashlib
import subprocess


def code_provenance(data_kind="synthetic"):
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    def git(*args):
        try:
            p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
            return p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"code_commit": git("rev-parse", "HEAD"), "working_tree_dirty": bool(status) if status is not None else None,
            "source_tree_sha256": digest.hexdigest(), "data_kind": data_kind}
