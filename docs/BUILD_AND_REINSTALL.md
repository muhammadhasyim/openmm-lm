# Build and Install OpenMM with Pixi

OpenMM uses [Pixi](https://pixi.prefix.dev/latest/) as the primary build and environment manager. Pixi provides reproducible conda-forge toolchains (compilers, CUDA, SWIG, Python) and drives the existing CMake build via the `pixi-build-cmake` backend.

## Prerequisites

- [Pixi](https://pixi.prefix.dev/latest/#installation) installed (`curl -fsSL https://pixi.sh/install.sh | sh`)
- Linux: NVIDIA driver for CUDA builds (toolkit comes from conda-forge)
- macOS / Windows: CPU + Reference platforms only (no CUDA)

## Quick start

From the repository root:

```bash
# Resolve environment and build+install openmm from source
pixi install

# Verify installation
pixi run smoke
pixi run info
```

The first `pixi install` builds the `openmm` conda package from this repo and installs it into `.pixi/envs/default`.

## Environments

| Environment | Command | Purpose |
|-------------|---------|---------|
| `default` | `pixi install` | Core OpenMM build |
| `test` | `pixi install -e test` | Adds pytest; run C++/Python tests |
| `ml` | `pixi install -e ml` | Adds fairchem-core + PyTorch for ML potentials |
| `docs` | `pixi install -e docs` | Sphinx + Doxygen |

## Common tasks

```bash
# Build the conda package artifact (.conda)
pixi build

# Run Python wrapper tests
pixi run -e test test-py

# Run C++ tests (CTest via devtools/run-ctest.py)
pixi run -e test test-cpp

# ML stack smoke test
pixi install -e ml
pixi run -e ml smoke-ml
```

## Platform behavior

Pixi target configuration in [`pixi.toml`](../pixi.toml) sets platform-specific CMake flags:

- **linux-64 / win-64**: CUDA + OpenCL + CPU plugins; CUDA toolkit from conda-forge
- **osx-64 / osx-arm64**: CPU + Reference only (CUDA disabled)

## Plugin directory

After install, platform plugins live at:

```bash
export OPENMM_PLUGIN_DIR="$PIXI_PROJECT_ROOT/.pixi/envs/default/lib/plugins"
```

Pixi sets `CONDA_PREFIX` inside activated environments; plugins are under `$CONDA_PREFIX/lib/plugins`.

## HuggingFace (UMA models)

UMA models are gated. Log in once before using ML potentials:

```bash
pixi run -e ml huggingface-cli login
```

## How it works

1. **`pixi.toml`** declares workspace dependencies and the `openmm` package build.
2. **`pixi-build-cmake`** runs `cmake -GNinja -DCMAKE_INSTALL_PREFIX=$PREFIX` with OpenMM-specific flags.
3. **`ninja`** builds `libOpenMM`, platform plugins, and the SWIG Python module.
4. **`ninja install`** installs C++ libs/headers/plugins; with `-DOPENMM_PYTHON_INSTALL_TO_SITEPACKAGES=ON`, the Python module is installed via pip into the same prefix.

## Legacy manual build

The old conda + cmake workflow is deprecated. If you must build manually:

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release \
         -DOPENMM_BUILD_CUDA_LIB=ON -DOPENMM_BUILD_PYTHON_WRAPPERS=ON
ninja -j$(nproc) && ninja install && ninja PythonInstall
```

Prefer `pixi install` for reproducible builds.

## Python-only changes (no rebuild)

If you only modified Python files under `wrappers/python/openmm/` or `openmmml/`:

```bash
cp wrappers/python/openmm/cavitymd/*.py \
   .pixi/envs/default/lib/python*/site-packages/openmm/cavitymd/
```

Or re-run `pixi install` to rebuild the package.

## CUDA: `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (222)

Pixi builds link the CUDA plugin against conda-forge **NVRTC 13.x**. OpenMM JIT-compiles CUDA kernels at runtime via NVRTC; if the PTX version exceeds what your NVIDIA driver supports (check `nvidia-smi` → “CUDA Version”), the CUDA platform fails with error 222 while Reference and CPU still work.

**Fix** (Linux with a system CUDA toolkit, e.g. `/usr/local/cuda`):

```bash
pixi run -e test fix-cuda
pixi run -e test smoke
```

This rebuilds `libOpenMMCUDA.so` linked against your system `libnvrtc` (typically 12.x) and installs it into the active Pixi environment. Override the toolkit path with `OPENMM_NVRTC_ROOT=/path/to/cuda` if needed.

Alternatively, update the NVIDIA driver to one that supports the conda CUDA toolkit version.

## Legacy install scripts and CI

The following are **deprecated** for local development:

- `scripts/install_openmm_fairchem_base.sh` — exits with an error; use `pixi install`
- `scripts/build_and_test.sh` — exits with an error; use `pixi run -e test test-py` and `test-cpp`

Upstream-style matrix CI (CUDA versions, OpenCL, Windows) remains in `.github/workflows/CI.yml` as a **nightly / manual** workflow. Day-to-day PR CI is `.github/workflows/pixi-ci.yml`.

For CUDA validation on a machine with an NVIDIA GPU, trigger the optional **Pixi linux-64 CUDA smoke (GPU runner)** job in `pixi-ci.yml` via **workflow_dispatch** on a self-hosted runner labeled `self-hosted`, `linux`, and `cuda`.
