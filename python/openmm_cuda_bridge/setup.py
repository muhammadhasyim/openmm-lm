from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parent
OPENMM_DIR = Path(os.environ.get("OPENMM_DIR", ROOT))
DEFAULT_OPENMM_LIB_DIR = OPENMM_DIR / "build"
if not (DEFAULT_OPENMM_LIB_DIR / "libOpenMM.so").exists():
    DEFAULT_OPENMM_LIB_DIR = OPENMM_DIR / "lib"
OPENMM_LIB_DIR = Path(os.environ.get("OPENMM_LIB_DIR", DEFAULT_OPENMM_LIB_DIR))
OPENMM_CUDA_LIB_DIR = Path(
    os.environ.get("OPENMM_CUDA_LIB_DIR", OPENMM_LIB_DIR)
)
if not (OPENMM_CUDA_LIB_DIR / "libOpenMMCUDA.so").exists():
    plugins_dir = OPENMM_LIB_DIR / "plugins"
    if (plugins_dir / "libOpenMMCUDA.so").exists():
        OPENMM_CUDA_LIB_DIR = plugins_dir
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
CUDA_LIB_DIR = CUDA_HOME / "lib64"
CUDA_DRIVER_LIB_DIR = CUDA_LIB_DIR
if not (CUDA_DRIVER_LIB_DIR / "libcuda.so").exists() and (CUDA_LIB_DIR / "stubs" / "libcuda.so").exists():
    CUDA_DRIVER_LIB_DIR = CUDA_LIB_DIR / "stubs"

KERNEL_SRC = PKG_ROOT / "src" / "bridge_kernels.cu"
KERNEL_OBJ = PKG_ROOT / "build" / "bridge_kernels.o"

include_dirs = [
    str(OPENMM_DIR / "openmmapi" / "include"),
    str(OPENMM_DIR / "olla" / "include"),
    str(OPENMM_DIR / "serialization" / "include"),
    str(OPENMM_DIR / "libraries" / "lepton" / "include"),
    str(OPENMM_DIR / "platforms" / "common" / "include"),
    str(OPENMM_DIR / "platforms" / "cuda" / "include"),
    str(OPENMM_DIR / "platforms" / "cuda" / "src"),
    str(CUDA_HOME / "include"),
]

library_dirs = [
    str(OPENMM_LIB_DIR),
    str(OPENMM_CUDA_LIB_DIR),
    str(CUDA_LIB_DIR),
    str(CUDA_DRIVER_LIB_DIR),
]


def compile_cuda_kernels() -> Path:
    KERNEL_OBJ.parent.mkdir(parents=True, exist_ok=True)
    nvcc = CUDA_HOME / "bin" / "nvcc"
    cmd = [
        str(nvcc),
        "-c",
        str(KERNEL_SRC),
        "-o",
        str(KERNEL_OBJ),
        "-std=c++17",
        "--expt-relaxed-constexpr",
        f"-I{CUDA_HOME / 'include'}",
        "-Xcompiler",
        "-fPIC",
    ]
    subprocess.check_call(cmd)
    return KERNEL_OBJ


class BuildBridgeExt(BuildExtension):
    def build_extensions(self) -> None:
        compile_cuda_kernels()
        super().build_extensions()


setup(
    name="openmm-cuda-bridge",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            "openmm_cuda_bridge._openmm_cuda_bridge",
            ["src/openmm_cuda_bridge.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[
                str(OPENMM_LIB_DIR),
                str(OPENMM_CUDA_LIB_DIR),
                str(CUDA_LIB_DIR),
            ],
            extra_objects=[str(KERNEL_OBJ)],
            libraries=["OpenMM", "OpenMMCUDA", "cuda", "cudart"],
            extra_compile_args=["-std=c++17"],
            extra_link_args=[
                f"-Wl,-rpath,{OPENMM_LIB_DIR}",
                f"-Wl,-rpath,{OPENMM_CUDA_LIB_DIR}",
                f"-Wl,-rpath,{CUDA_HOME / 'lib64'}",
            ],
        )
    ],
    cmdclass={"build_ext": BuildBridgeExt},
)
