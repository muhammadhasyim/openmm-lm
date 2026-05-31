# Deprecated: conda environment files for legacy CI

These YAML files were used by the pre-Pixi GitHub Actions workflow
(`.github/workflows/CI.yml`, now **nightly / manual only**) with
`conda-incubator/setup-miniconda`. They are **superseded by [`pixi.toml`](../../../pixi.toml)**.

For local development and PR CI, use:

```bash
pixi install          # default build
pixi install -e test  # build + pytest + C++ tests
pixi install -e ml    # build + ML deps
```

Primary CI: [`.github/workflows/pixi-ci.yml`](../../../.github/workflows/pixi-ci.yml).
