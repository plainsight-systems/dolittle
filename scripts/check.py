#!/usr/bin/env python3
"""Build/check entry point; source application launch is scripts/run.py.

Host units need CMake, a C++20 compiler and OpenSSL 3 development files.
CUDA source builds default to the digest-pinned Docker environment. GPU checks
can execute those binaries on a compatible Linux host, or inside that environment
when the NVIDIA Container Toolkit is configured. Models/fixtures stay outside
the image. No downloads or administrator changes are performed by GPU checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tidy
import formatting

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "dolittle-build:cuda13.2"


def command(args, **kwargs):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True, **kwargs)


def container_command(args, gpu=False, fixtures=None, force_ptx=False, image=IMAGE, read_roots=()):
    result = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
              "--volume", f"{ROOT}:{ROOT}", "--workdir", ROOT]
    if gpu:
        result += ["--gpus", "all"]
    if fixtures and not fixtures.is_relative_to(ROOT):
        result += ["--volume", f"{fixtures}:{fixtures}:ro"]
    for root in sorted(set(read_roots)):
        if not root.is_relative_to(ROOT) and root != fixtures:
            result += ["--volume", f"{root}:{root}:ro"]
    if force_ptx:
        result += ["--env", "CUDA_FORCE_PTX_JIT=1"]
    return result + [image] + list(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    formatting.add_arguments(sub.add_parser("format", help="check Google C/C++/CUDA formatting; --fix applies it"))
    tidy.add_arguments(sub.add_parser("tidy", help="run pinned clang-tidy static analysis; findings are errors"))
    host = sub.add_parser("host", help="build and run isolated C++ units")
    host.add_argument("--build-dir", type=Path, default=ROOT / "build/host")
    build = sub.add_parser("cuda-build", help="build CUDA engine and integration drivers")
    build.add_argument("--native", action="store_true", help="use installed CUDA 13.2+ and build dependencies")
    build.add_argument("--build-dir", type=Path, default=ROOT / "build/cuda")
    gpu = sub.add_parser("gpu", help="run device, tiny and pinned 8B target-to-AV scenarios")
    gpu.add_argument("--fixtures", type=Path, required=True)
    gpu.add_argument("--build-dir", type=Path, default=ROOT / "build/cuda")
    gpu.add_argument("--container", action="store_true", help="requires working Docker GPU access")
    gpu.add_argument("--force-ptx", action="store_true")
    gpu.add_argument("--report-dir", type=Path, default=ROOT / ".cache/checks")
    args = parser.parse_args()
    if args.action == "format":
        try:
            raise SystemExit(formatting.run(args))
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            parser.exit(2, f"format: {error}\n")
    if args.action == "tidy":
        try:
            if args.container:
                if args.clang_tidy:
                    parser.error("--clang-tidy selects a host tool; omit it with --container")
                if not args.build_dir.resolve().is_relative_to(ROOT) or not args.report_dir.resolve().is_relative_to(ROOT):
                    parser.error("container build/report directories must be inside the checkout")
                command(["docker", "build", "--target", "analysis", "--tag", tidy.IMAGE, ROOT])
                command(container_command(["python3", ROOT / "scripts/check.py", "tidy",
                    "--build-dir", args.build_dir.resolve(), "--report-dir", args.report_dir.resolve(),
                    "--jobs", str(args.jobs)], image=tidy.IMAGE))
                return
            raise SystemExit(tidy.run(args))
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            parser.exit(2, f"clang-tidy: {error}\n")
    build_dir = args.build_dir.resolve()
    if args.action in {"host", "cuda-build"}:
        configure = ["cmake", "-S", ROOT, "-B", build_dir,
                     f"-DDOLITTLE_CUDA={'ON' if args.action == 'cuda-build' else 'OFF'}",
                     f"-DCMAKE_BUILD_TYPE={'Release' if args.action == 'cuda-build' else 'Debug'}"]
        compile_command = ["cmake", "--build", build_dir, "--parallel"]
        if args.action == "cuda-build" and not args.native:
            command(["docker", "build", "--tag", IMAGE, ROOT])
            command(container_command(configure))
            command(container_command(compile_command))
        else:
            command(configure)
            command(compile_command)
        if args.action == "host":
            command(["ctest", "--test-dir", build_dir, "--label-regex", "^unit$", "--output-on-failure"])
    else:
        fixtures = args.fixtures.resolve(strict=True)
        av_fixtures = fixtures.parent / "av-fixtures"
        read_roots = [av_fixtures]
        for kind in ("tiny-fp32", "tiny-bf16", "qwen3-8b-bf16"):
            reference = json.loads((av_fixtures / f"{kind}.json").read_text())
            for source, expected in (
                (fixtures / kind / "fixture.json", reference["target_fixture_sha256"]),
                (ROOT / "scripts/validation/av_reference.py", reference["generator_sha256"]),
                (ROOT / "scripts/validation/forward_reference.py", reference["oracle_sha256"]),
            ):
                with source.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != expected:
                    parser.error(f"frozen AV provenance mismatch: {source}")
            read_roots.append(Path(reference["root"]).resolve(strict=True))
        args.report_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        if args.force_ptx:
            environment["CUDA_FORCE_PTX_JIT"] = "1"
        else:
            environment.pop("CUDA_FORCE_PTX_JIT", None)
        suffix = "ptx" if args.force_ptx else "native"
        scenarios = {
            "device": [build_dir / "test_device"],
            "generation-tiny": [build_dir / "test_generation", fixtures / "tiny-fp32/fixture.json",
                             fixtures / "tiny-bf16/fixture.json"],
            "generation-8b": [build_dir / "test_generation", fixtures / "qwen3-8b-bf16/fixture.json"],
        }
        for name, invocation in scenarios.items():
            if args.container:
                invocation = container_command(invocation, gpu=True, fixtures=fixtures, force_ptx=args.force_ptx,
                                               read_roots=read_roots)
            report = args.report_dir / f"{name}-{suffix}.json"
            print(f"Running {name} ({suffix}); report: {report}", flush=True)
            with report.open("w") as output:
                command(invocation, stdout=output, env=environment)


if __name__ == "__main__":
    main()
