from __future__ import annotations

import sys

from pybind11 import get_include
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2", "/std:c++17"],
        "unix": ["-O3", "-std=c++17", "-fvisibility=hidden"],
    }
    l_opts = {"msvc": [], "unix": []}

    def build_extensions(self) -> None:
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            ext.extra_compile_args = self.c_opts.get(compiler_type, [])
            ext.extra_link_args = self.l_opts.get(compiler_type, [])
            if compiler_type == "unix" and sys.platform == "darwin":
                ext.extra_compile_args += ["-mmacosx-version-min=10.15"]
                ext.extra_link_args += ["-mmacosx-version-min=10.15"]
        super().build_extensions()


extensions = [
    Extension(
        "openvad._core",
        ["native/vad_core.cpp"],
        include_dirs=[get_include()],
        language="c++",
    )
]


setup(ext_modules=extensions, cmdclass={"build_ext": BuildExt})
