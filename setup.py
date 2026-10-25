"""Build configuration for Migdal and the optional radial Cython extension."""

from __future__ import annotations

import os
import importlib.machinery
import shutil
import sys
import tempfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _env_truthy(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _split_env_flags(name):
    raw = os.environ.get(name)
    if not raw:
        return None
    return raw.split()


def _openmp_candidates(compiler_type):
    compile_override = _split_env_flags("MIGDAL_OPENMP_COMPILE_ARGS")
    link_override = _split_env_flags("MIGDAL_OPENMP_LINK_ARGS")
    if compile_override is not None or link_override is not None:
        flags = compile_override if compile_override is not None else []
        return [(flags, link_override if link_override is not None else flags)]

    if compiler_type == "msvc":
        return [(["/openmp"], [])]
    if sys.platform == "darwin":
        return [
            (["-Xpreprocessor", "-fopenmp"], ["-lomp"]),
            (["-fopenmp"], ["-fopenmp"]),
        ]
    return [
        (["-fopenmp"], ["-fopenmp"]),
        (["-qopenmp"], ["-qopenmp"]),
        (["-openmp"], ["-openmp"]),
    ]


def _remove_inplace_radial_artifacts(source_dir=None):
    if source_dir is None:
        source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migdal")
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    prefixes = ("_radial_kernels", "_qw_radial_kernels")
    removed = []
    for name in os.listdir(source_dir):
        if not name.endswith(suffixes) or not name.startswith(prefixes):
            continue
        path = os.path.join(source_dir, name)
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        removed.append(path)
    return removed


class OptionalOpenMPBuildExt(build_ext):
    """Build the optimized radial extension only when OpenMP is available."""

    def run(self):
        if cythonize is None:
            self._skip_radial_extension(
                "Cython is not available; skipping migdal radial Cython extensions. "
                "The package will use the pure-Python radial fallback."
            )
            return
        return super().run()

    def build_extensions(self):
        if _env_truthy("MIGDAL_DISABLE_OPENMP"):
            self._skip_radial_extension(
                "MIGDAL_DISABLE_OPENMP is set; skipping radial Cython extension"
            )
            return

        for compile_args, link_args in _openmp_candidates(self.compiler.compiler_type):
            if self._can_build_openmp(compile_args, link_args):
                for ext in self.extensions:
                    ext.extra_compile_args = (
                        list(ext.extra_compile_args or []) + compile_args
                    )
                    ext.extra_link_args = list(ext.extra_link_args or []) + link_args
                return super().build_extensions()

        message = (
            "OpenMP probe failed; skipping migdal radial Cython extensions. "
            "The package will use the pure-Python radial fallback."
        )
        if _env_truthy("MIGDAL_REQUIRE_OPENMP"):
            raise RuntimeError(message)
        self._skip_radial_extension(message)

    def _skip_radial_extension(self, message):
        self.warn(message)
        removed = _remove_inplace_radial_artifacts()
        if removed:
            names = ", ".join(os.path.basename(path) for path in removed)
            self.warn(f"removed stale radial extension artifact(s): {names}")
        self.extensions = []

    def _can_build_openmp(self, compile_args, link_args):
        tmpdir = tempfile.mkdtemp(prefix="migdal-openmp-")
        source = os.path.join(tmpdir, "probe.c")
        exe = os.path.join(tmpdir, "probe")
        try:
            with open(source, "w", encoding="ascii") as handle:
                handle.write(
                    "#include <omp.h>\n"
                    "int main(void) { return omp_get_max_threads() < 1; }\n"
                )
            objects = self.compiler.compile(
                [source], output_dir=tmpdir, extra_postargs=compile_args
            )
            self.compiler.link_executable(objects, exe, extra_postargs=link_args)
            return True
        except Exception:
            return False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover - exercised only by build tooling
    if _env_truthy("MIGDAL_REQUIRE_OPENMP"):
        raise SystemExit(
            "Cython is required to build migdal radial kernels."
        ) from exc
    cythonize = None


extensions = [
    Extension(
        "migdal._radial_kernels",
        ["migdal/_radial_kernels.pyx"],
    ),
    Extension(
        "migdal._qw_radial_kernels",
        ["migdal/_qw_radial_kernels.pyx"],
    ),
]


setup(
    name="migdal",
    packages=["migdal"],
    ext_modules=(
        []
        if cythonize is None
        else cythonize(extensions, compiler_directives={"language_level": "3"})
    ),
    cmdclass={"build_ext": OptionalOpenMPBuildExt},
)
