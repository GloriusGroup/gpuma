# GPUMA

<div align="center">
  <img src="docs/logo_bg.png" alt="GPUMA Logo"/>
</div>

---

GPUMA is a minimalist Python toolkit for facile and rapid high-throughput molecular geometry optimization 
based on the [UMA/OMol25 machine-learning interatomic potential](https://arxiv.org/abs/2505.08762).  

GPUMA is especially designed for batch optimizations of many structures (conformer ensembles, datasets) on GPU,
ensuring efficient parallelization and maximum GPU utilization by leveraging the [torch-sim library](https://arxiv.org/abs/2508.06628).
It wraps Fairchem UMA models and torch-sim functionality to provide both a simple command-line 
interface (CLI) and a small but expressive Python API for single- and multi-structure optimizations.

If conformer sampling is desired, GPUMA can generate conformer ensembles on the fly from SMILES strings 
using the [morfeus library](https://digital-chemistry-laboratory.github.io/morfeus/). Alternative input formats
are described in the CLI section below.

Feedback and improvements are always welcome!

## Installation

### Option 1: Install from PyPI (recommended)

```bash
pip install gpuma
```

This installs `gpuma` together with its core dependencies. Make sure you are using
Python 3.12 or newer.

> ⚠️ **Required for UMA models:**</br>
> To access the UMA models on Hugging Face, **you must provide a token** either via the `HUGGINGFACE_TOKEN` environment variable or via the config (direct token string or path to a file containing the token).

### Option 2: Install from source

```bash
# clone the repository
git clone https://github.com/niklashoelter/gpuma.git
cd gpuma

# install using (uv) pip
uv pip install .
# or, without uv:
pip install .
```

## Documentation

Full documentation is available at [https://niklashoelter.github.io/gpuma/](https://niklashoelter.github.io/gpuma/).

For local browsing of the Markdown sources, see in particular:
- [docs/index.md](docs/index.md) – overview and getting started
- [docs/install.md](docs/install.md) – installation details
- [docs/cli.md](docs/cli.md) – CLI options and input formats
- [docs/config.md](docs/config.md) – configuration file schema and examples
- [docs/reference.md](docs/reference.md) – API and configuration reference

Using a configuration file is highly recommended for reproducibility and ease of use.

Also check the [examples/](examples) folder in the repository for sample config files and usage examples:
- [examples/config.json](examples/config.json) – minimal example configuration
- [examples/example_single_optimization.py](examples/example_single_optimization.py) – single-structure optimization from Python
- [examples/example_ensemble_optimization.py](examples/example_ensemble_optimization.py) – ensemble/multi-structure optimization from Python

## CLI Usage

The CLI is provided via the command `gpuma`. For best results, create a
config file (JSON or YAML) and reference it in all CLI calls (see [examples/config.json](examples/config.json) for a minimal example).

### Examples: Batch optimization of multiple XYZ structures

Optimize all XYZ files in a directory (each file containing a single structure):

```bash
gpuma optimize --config examples/config.json --xyz-dir examples/example_input_xyzs/multi_xyz_dir/
```

Optimize multiple structures contained in a single multi-XYZ file:

```bash
gpuma optimize --config examples/config.json --xyz examples/example_input_xyzs/multi_xyz_file.xyz
```

Refer to the [CLI documentation](docs/cli.md) for details on configuration options, supported input formats (SMILES, XYZ, directories, multi-XYZ files), and additional CLI examples.

## Python API

A minimalistic and high-level Python API is provided for easy integration into custom scripts and workflows.

For example usage, see:
- [examples/example_single_optimization.py](examples/example_single_optimization.py)
- [examples/example_ensemble_optimization.py](examples/example_ensemble_optimization.py)

Please refer to the documentation and examples for detailed usage examples and API reference.

## Known limitations

When a run is started from SMILES, an RDKit force field (via the morfeus library) is used to generate an initial structure. Spin is not taken into account during this step, so the initial estimated geometries can be incorrect. When the UMA/Omol25 models are applied subsequently, the structure can sometimes be optimized to a maximum rather than a minimum because the model is not provided with Hessian matrices. This behavior only affects runs originating from SMILES; it does not occur with better starting geometries (e.g., when starting from XYZ files).

## Troubleshooting
- Missing libraries: install optional dependencies like `pyyaml` if you use YAML configs.
- Fairchem/UMA: ensure network access for model downloads and optionally set or provide 
`huggingface_token` (e.g., via a token file) to access the UMA model family.

## License
MIT License (see LICENSE)
