# Installation

## Option 1: Install from PyPI (recommended)

```bash
pip install gpuma
```

This installs `gpuma` together with its core dependencies. At the moment, installation and tests have only been
validated under Python 3.12; using other Python versions is currently
considered experimental.


- **Using a `uv` virtual environment**
  ```powershell
  # create and activate a fresh environment
  uv venv .venv

  # activate the environment

  # install gpuma from PyPI inside the environment
  uv pip install gpuma
  ```

- **Using a `conda` environment**
  ```powershell
  # create and activate a fresh environment with Python 3.12
  conda create -n gpuma-py312 python=3.12
  conda activate gpuma-py312

  # install gpuma from PyPI inside the environment
  pip install gpuma
  ```

> ⚠️ **Required for UMA models:**</br>
> To access the UMA models on Hugging Face, **you must provide a token** either via the `HUGGINGFACE_TOKEN` environment variable or via the config (direct token string or path to a file containing the token).

## Option 2: Install from source

```bash
# clone the repository
git clone https://github.com/niklashoelter/gpuma.git
cd gpuma

# install using (uv) pip
uv pip install .
# or, without uv:
pip install .
```
