"""Verify, then build the rubric's Option A submission without caches or secrets."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PYTEST_ADDOPTS="--basetemp=.pytest_package_tmp")
    subprocess.run([sys.executable, "scripts/verify.py"], cwd=ROOT, env=env, check=True)
    paths = []
    for folder in ("submission", "results", "notebooks", "colab", "tests", "src", "scripts", "data", "docs"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.name != ".gitkeep")
    paths.extend(p for p in (ROOT / "adapters" / "correct").iterdir() if p.is_file())
    paths.extend(ROOT / name for name in ("requirements.txt", "requirements-cpu.txt",
        "pyproject.toml", "README.md", "rubric.md", "BONUS-CHALLENGE.md",
        "BONUS-CHALLENGE-EN.md", "HARDWARE-GUIDE.md", "SIMULATION-FINDINGS.md", "LICENSE", "Makefile",
        ".gitattributes", ".env.example"))
    pinned = ROOT / "submission" / "requirements-lock.txt"
    if pinned.exists() and pinned not in paths:
        paths.append(pinned)
    archive = ROOT / "submission" / "lab21_2A202602774.zip"
    paths = sorted(set(p for p in paths if p.suffix != ".zip"))
    manifest = {str(p.relative_to(ROOT)).replace("\\", "/"):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, f"lab21_2A202602774/{p.relative_to(ROOT).as_posix()}")
        z.writestr("lab21_2A202602774/MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"Submission: {archive} ({archive.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
