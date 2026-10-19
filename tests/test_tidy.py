"""In-memory contracts for compile-command adaptation; no compiler or GPU runs."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tidy import clang_arguments


class CompileCommandTests(unittest.TestCase):
    def test_host_semantics_are_preserved(self):
        args = ["/usr/bin/c++", "-std=c++20", "-Iinclude with spaces", "-DMODE=1", "-Werror", "-c", "unit.cpp"]
        self.assertEqual(clang_arguments(args, Path(".")), args)

    def test_cuda_semantics_survive_driver_translation(self):
        args = ["/toolkit/bin/nvcc", "-forward-unknown-to-host-compiler",
                "-std=c++20", "-Iinclude with spaces", "-DMODE=1",
                "--generate-code=arch=compute_86,code=[sm_86]",
                "--generate-code=arch=compute_86,code=[compute_86]",
                "--fmad=false", "--Werror=all-warnings", "-Xcompiler=-Wall,-Wextra,-Werror",
                "-x", "cu", "-c", "kernel.cu"]
        result = clang_arguments(args, Path("."))
        for option in ("-std=c++20", "-Iinclude with spaces", "-DMODE=1", "-Werror",
                       "-Wall", "-Wextra", "-ffp-contract=off", "--cuda-path=/toolkit",
                       "--cuda-gpu-arch=sm_86", "kernel.cu"):
            self.assertIn(option, result)
        self.assertEqual(result[result.index("-x") + 1], "cuda")

    def test_unknown_options_are_not_silently_dropped(self):
        args = ["/toolkit/bin/nvcc", "--future-semantic-option", "--generate-code=arch=compute_86,code=[sm_86]"]
        self.assertIn("--future-semantic-option", clang_arguments(args, Path(".")))

    def test_ambiguous_or_missing_architecture_is_rejected(self):
        for flags in ([], ["--generate-code=broken"],
                      ["--generate-code=arch=compute_86,code=[sm_86]",
                       "--generate-code=arch=compute_90,code=[sm_90]"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                clang_arguments(["/toolkit/bin/nvcc", *flags], Path("."))


if __name__ == "__main__":
    unittest.main()
