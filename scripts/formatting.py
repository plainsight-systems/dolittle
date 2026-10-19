"""Google-style formatting for project C/C++/CUDA sources, using one pinned tool.

The editor wrapper and check command share discovery. Never install on save or
silently use an incompatible bundled formatter. Check mode never changes files.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = "21.1.8"
SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh"}


def find_tool():
    requested = os.environ.get("CLANG_FORMAT")
    candidates = [requested] if requested else [
        str(ROOT / ".venv-tools/bin/clang-format"), "clang-format-21", "clang-format",
    ]
    if not requested and sys.platform == "darwin" and shutil.which("brew"):
        prefix = subprocess.run(["brew", "--prefix", "llvm@21"], capture_output=True, text=True)
        if prefix.returncode == 0:
            candidates.append(str(Path(prefix.stdout.strip()) / "bin/clang-format"))
    for candidate in candidates:
        binary = shutil.which(candidate)
        if not binary or Path(binary).resolve() == ROOT / "scripts/clang-format":
            continue
        version = subprocess.run([binary, "--version"], stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True)
        if version.returncode == 0 and re.search(r"\bversion " + re.escape(VERSION) + r"\b", version.stdout):
            return str(Path(binary).absolute())
    raise ValueError(f"clang-format {VERSION} is required. Run: python3 -m venv .venv-tools && "
                     ".venv-tools/bin/python -m pip install -r scripts/requirements-tools.txt")


def add_arguments(parser):
    parser.add_argument("--fix", action="store_true", help="format files in place; default is read-only checking")


def run(args):
    binary = find_tool()
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "src", "tests"],
        cwd=ROOT,
    )
    files = sorted({os.fsdecode(name) for name in listed.split(b"\0") if name
                    and Path(os.fsdecode(name)).suffix in SUFFIXES
                    and (ROOT / os.fsdecode(name)).is_file()})
    if not files:
        raise ValueError("No project C/C++/CUDA files found")
    flags = ["-i"] if args.fix else ["--dry-run", "--Werror"]
    result = subprocess.run([binary, "--style=file", *flags, *files], cwd=ROOT)
    if result.returncode == 0:
        print(f"clang-format {VERSION}: {len(files)} files {'formatted' if args.fix else 'clean'} (Google)")
    return result.returncode
