# Install OpenMM (Pixi)

The legacy `install_openmm_fairchem_base.sh` script is **deprecated**. Use [Pixi](https://pixi.prefix.dev/latest/) from the repository root.

## Quick start

```bash
pixi install              # build + install OpenMM
pixi run smoke            # verify platforms (Reference, CPU, CUDA on Linux)
```

## Test and ML environments

```bash
pixi install -e test
pixi run -e test smoke
pixi run -e test test-py
pixi run -e test test-cpp

# Optional ML stack (FairChem, PyTorch)
pixi install -e ml
pixi run -e ml install-ml
pixi run -e ml smoke-ml
```

## Linux CUDA (error 222)

If the CUDA platform fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` and a system CUDA toolkit is installed (e.g. `/usr/local/cuda`):

```bash
pixi run -e test fix-cuda
pixi run -e test smoke
```

See [docs/BUILD_AND_REINSTALL.md](../docs/BUILD_AND_REINSTALL.md) for details.

## Legacy script

`scripts/install_openmm_fairchem_base.sh` now exits with an error and prints Pixi instructions. Do not use it for new installs.
