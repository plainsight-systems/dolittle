#!/usr/bin/env python3
"""Run pinned clang-tidy against project entries in a CMake compilation database.

Analysis uses a temporary copy of the commands. NVCC-only driver options are
translated for Clang; production commands and source files are never rewritten.
Headers are checked through their translation units. Reports stay in .local/.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LLVM_VERSION = "21.1.8" if sys.platform == "darwin" else "22.1.8"
IMAGE = "dolittle-analysis:llvm22.1.8"


def add_arguments(parser):
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/host-debug")
    parser.add_argument("--clang-tidy", default=os.environ.get("CLANG_TIDY"))
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--report-dir", type=Path, default=ROOT / ".local/clang-tidy")
    parser.add_argument("--container", action="store_true",
                        help="build/use the pinned Linux analysis image; requires a configured CUDA build")


def find_tool(requested):
    candidates = [requested] if requested else ["clang-tidy-" + LLVM_VERSION.split(".")[0], "clang-tidy"]
    if not requested and sys.platform == "darwin" and shutil.which("brew"):
        prefix = subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True)
        if prefix.returncode == 0:
            candidates.append(str(Path(prefix.stdout.strip()) / "bin/clang-tidy"))
    for candidate in candidates:
        binary = shutil.which(candidate)
        if binary:
            version = subprocess.check_output([binary, "--version"], text=True)
            if re.search(r"\bversion " + re.escape(LLVM_VERSION) + r"\b", version):
                return binary
    raise ValueError(f"clang-tidy {LLVM_VERSION} is required; set CLANG_TIDY or --clang-tidy, "
                     "or use --container on the Linux CUDA checkout")


def expand_responses(arguments, directory, seen=frozenset()):
    """Expand Clang @files and NVCC --options-file, including nested responses."""
    result = []
    iterator = iter(arguments)
    for arg in iterator:
        files = None
        if arg in {"--options-file", "-optf"}:
            files = next(iterator, "").split(",")
        elif arg.startswith("--options-file="):
            files = arg.split("=", 1)[1].split(",")
        elif arg.startswith("@"):
            files = [arg[1:]]
        if files is None:
            result.append(arg)
            continue
        for name in files:
            path = (directory / name).resolve()
            if not name or path in seen:
                raise ValueError(f"Missing or recursive response file: {path}")
            result.extend(expand_responses(shlex.split(path.read_text()), directory, seen | {path}))
    return result


def clang_arguments(arguments, directory):
    """Preserve semantic flags; translate only the supported NVCC driver flags.

    Unknown options remain visible to Clang and fail analysis instead of silently
    dropping a define, include, architecture, or numerical setting.
    """
    compiler, *flags = arguments
    flags = expand_responses(flags, directory)
    if Path(compiler).name != "nvcc":
        return [compiler, *flags]
    result = ["clang++", "--cuda-path=" + str(Path(compiler).parent.parent),
              "--cuda-host-only"]
    architectures = set()
    iterator = iter(flags)
    for flag in iterator:
        if flag in {"-forward-unknown-to-host-compiler", "--forward-unknown-to-host-compiler"}:
            continue
        if flag.startswith("--generate-code="):
            match = re.search(r"arch=compute_(\d+)", flag)
            if not match:
                raise ValueError(f"Unsupported CUDA architecture option: {flag}")
            architectures.add(match[1])
        elif flag == "--fmad=false":
            result.append("-ffp-contract=off")
        elif flag == "--Werror=all-warnings":
            result.append("-Werror")
        elif flag.startswith("-Xcompiler="):
            result.extend(flag.split("=", 1)[1].split(","))
        elif flag == "-x":
            language = next(iterator)
            result.extend(["-x", "cuda" if language == "cu" else language])
        else:
            result.append(flag)
    if len(architectures) != 1:
        raise ValueError(f"Expected one CUDA baseline architecture, found {sorted(architectures)}")
    result.append("--cuda-gpu-arch=sm_" + next(iter(architectures)))
    return result


def macos_arguments(arguments):
    if sys.platform != "darwin":
        return []
    # Respect an explicitly selected SDK in the build before using Xcode's default.
    sdk_name = None
    for index, flag in enumerate(arguments):
        if flag == "-isysroot":
            sdk_name = arguments[index + 1]
        elif flag.startswith("-isysroot"):
            sdk_name = flag[len("-isysroot"):]
        elif flag.startswith("--sysroot="):
            sdk_name = flag.split("=", 1)[1]
    sdk = Path(sdk_name or subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip())
    compiler = Path(subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip())
    # Newer Xcode SDKs carry libc++; older installations put it by the compiler.
    candidates = [sdk / "usr/include/c++/v1", compiler.parent.parent / "include/c++/v1"]
    headers = next((path for path in candidates if path.is_dir()), None)
    if headers is None:
        raise ValueError("The selected Xcode SDK/toolchain has no libc++ headers")
    return ["-isysroot", str(sdk), "-isystem", str(headers)]


def cuda_compatibility_arguments(arguments):
    cuda = next((flag.split("=", 1)[1] for flag in arguments if flag.startswith("--cuda-path=")), None)
    if cuda is None:
        return []
    header = (Path(cuda) / "include/cuda.h").read_text()
    version = re.search(r"^#define CUDA_VERSION\s+(\d+)", header, re.MULTILINE)
    if version is None or version[1] != "13020":
        raise ValueError("CUDA analysis is verified with toolkit 13.2; review Clang compatibility before using another SDK")
    # LLVM 22.1.8 includes the CUDA 13 texture-header and rsqrt declaration
    # fixes, but still declares 12.9 its latest partially supported SDK.
    # Ignore only that version notice, after enforcing the SDK checked above.
    # Parsing errors and all code diagnostics remain fatal. NVCC is unchanged.
    return ["-Wno-unknown-cuda-version"]


def run(args):
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    binary = find_tool(args.clang_tidy)
    build = args.build_dir.resolve()
    database = json.loads((build / "compile_commands.json").read_text())
    commands = []
    seen = set()
    for entry in database:
        directory = Path(entry["directory"]).resolve()
        source = (directory / entry["file"]).resolve()
        if not any(source.is_relative_to(ROOT / name) for name in ("src", "tests")):
            continue
        if source.suffix not in {".cpp", ".cu"}:
            continue
        if not source.is_file() or not directory.is_dir():
            raise ValueError(f"Stale compilation database entry: {source}; reconfigure CMake here")
        raw = entry.get("arguments") or shlex.split(entry["command"])
        arguments = clang_arguments(raw, directory)
        arguments += macos_arguments(arguments)
        arguments += cuda_compatibility_arguments(arguments)
        key = (str(source), tuple(arguments))
        if key in seen:
            continue
        seen.add(key)
        commands.append({"directory": str(directory), "file": str(source), "arguments": arguments})
        if source.suffix == ".cu":
            device_arguments = ["--cuda-device-only" if arg == "--cuda-host-only" else arg for arg in arguments]
            commands.append({"directory": str(directory), "file": str(source), "arguments": device_arguments})
    if not commands:
        raise ValueError("No project C++/CUDA entries found; reconfigure CMake in this checkout")
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    cuda_count = sum(entry["file"].endswith(".cu") for entry in commands)
    print(f"clang-tidy {LLVM_VERSION}: {len(commands)} compile commands "
          f"({cuda_count} CUDA); {'CUDA + host' if cuda_count else 'host only'}", flush=True)
    if cuda_count:
        print("CUDA 13.2 header compatibility mode: suppressing only Clang's SDK-version notice; "
              "host and device AST passes, no code generation.", flush=True)
    results = []
    # One database per command avoids running every duplicate configuration once
    # for each duplicate. Temporary analysis flags never replace CMake's database.
    with tempfile.TemporaryDirectory(prefix="commands-", dir=report) as temporary:
        def check(index_entry):
            index, entry = index_entry
            directory = Path(temporary) / str(index)
            directory.mkdir()
            (directory / "compile_commands.json").write_text(json.dumps([entry]))
            result = subprocess.run([binary, entry["file"], "-p", str(directory), "--quiet"],
                                    cwd=ROOT, capture_output=True, text=True)
            relative = str(Path(entry["file"]).relative_to(ROOT))
            log = report / f"{index:03d}-{Path(relative).name}.log"
            log.write_text(result.stdout + result.stderr)
            phase = "device" if "--cuda-device-only" in entry["arguments"] else "host"
            return {"file": relative, "phase": phase, "exit_code": result.returncode, "log": str(log)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for result in pool.map(check, enumerate(commands)):
                results.append(result)
                print(f"{'PASS' if result['exit_code'] == 0 else 'FAIL'} {result['file']} ({result['phase']})", flush=True)
                if result["exit_code"]:
                    print(Path(result["log"]).read_text(), flush=True)
    failed = sum(result["exit_code"] != 0 for result in results)
    (report / "summary.json").write_text(json.dumps({"llvm": LLVM_VERSION,
        "build": str(build), "cuda_commands": cuda_count, "failed": failed, "results": results}, indent=2) + "\n")
    print(f"{failed}/{len(results)} commands failed. Diagnostics: {report}", flush=True)
    return 1 if failed else 0
