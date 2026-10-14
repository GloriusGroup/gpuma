---
name: gpuma
description: Run gpuma to optimize molecular geometries on the GPU with machine-learned interatomic potentials (Fairchem UMA, ORB-v3, SevenNet), from SMILES, XYZ files or conformer ensembles. Use when asked to optimize structures, generate and relax conformers, pick an MLIP or optimizer, write a gpuma config, or debug a gpuma run.
---

# Running gpuma

gpuma relaxes many structures at once on one GPU through torch-sim. The CLI
is `gpuma`; the Python API is in `gpuma/__init__.py`. The authoritative
references are `docs/config.md` (every key and default) and `docs/cli.md`.

| Command | Input → output |
| --- | --- |
| `gpuma optimize --smiles S` / `--xyz F` | one structure → XYZ |
| `gpuma ensemble --smiles S --conformers N` | conformers of one molecule → multi-XYZ |
| `gpuma batch --multi-xyz F` / `--xyz-dir D` | many structures → multi-XYZ |
| `gpuma convert --smiles S` | SMILES → 3D, no optimization |
| `gpuma generate --smiles S --conformers N` | conformers, no optimization |
| `gpuma config --create F` / `--validate F` | write or check a config |

Every command takes `--output/-o`. All commands except `convert` and
`generate` take `--config/-c`.

## CLI traps

- **Global flags go before the subcommand.** `gpuma --device cuda:1 batch ...`
  works; `gpuma batch ... --device cuda:1` is an argparse error. The same
  holds for `--model-type`, `-v` and `-q`.
- **A missing config file is not an error.** Without `--config`, gpuma reads
  `./config.json`, and if that is absent it runs on built-in defaults: UMA
  (`uma-s-1p2`), which needs a Hugging Face token. Always pass `--config`.
- **`convert` and `generate` reject `--config`.** They read `./config.json`
  from the working directory, so run them from the config's directory or use
  the Python API (`smiles_to_xyz`, `smiles_to_ensemble`, which take `config=`).
- **`batch` and `optimize --xyz` ignore the config's `charge` and
  `multiplicity`.** The flags default to 0 and 1 and always win. Pass
  `--charge` and `--multiplicity` explicitly for anything not neutral and
  closed-shell. For SMILES input the charge comes from the SMILES.
- **`technical.device: "cuda:N"` for an index that does not exist falls back
  to `cuda:0` with only a warning.** On a shared machine check
  `nvidia-smi` first and pin the device, or set `CUDA_VISIBLE_DEVICES`.
- A failed command logs a traceback and exits 1.

## Config choices that matter

| Key | Default | Guidance |
| --- | --- | --- |
| `model.model_type` / `model_name` | `fairchem` / `uma-s-1p2` | For organic molecules use `orb` / `orb_v3_direct_omol`: no token, fast to load, and model choice changes conformer results little. UMA needs a token (`huggingface_token_file` or `HUGGINGFACE_TOKEN`) and its first load takes minutes. SevenNet multi-modal checkpoints (`7net-omni`) also need `model_modal`, e.g. `"omol25_high"`. |
| `model.d3_correction` | `false` | Only for SevenNet. UMA and ORB OMol models are trained on ωB97M-V, which already includes dispersion, so D3 counts it twice. |
| `optimization.batch_optimizer` | `fire` | Use `lbfgs`: 1.6-2.0× faster than FIRE and at least as accurate. Never `bfgs`, which is slower than FIRE. A misspelled value falls back to `fire` with only a log warning. |
| `optimization.batch_optimization_mode` | `batch` | Keep `batch` for more than a handful of structures. `sequential` runs ASE one structure at a time. |
| `optimization.force_convergence_criterion` | `5e-2` eV/Å | Tighten only if the geometry feeds something force-sensitive. |
| `conformer_generation.max_num_conformers` | `20` | Conformers generated per molecule. Raise it for flexible molecules. |
| `technical.max_memory_padding` | `0.95` | Lower it (e.g. `0.8`) after a CUDA OOM or on a GPU shared with other jobs. |
| `technical.memory_scales_with` | `auto` | Leave on `auto`. It picks `n_atoms` for UMA and `n_edges` for ORB and SevenNet. |

## Throughput

- **One process, many structures.** Every process pays model load and
  autobatcher calibration again. Put all structures into one multi-XYZ or
  directory and run `batch` once, never a loop of `optimize` calls.
- **ORB has a fixed ~30 s per process** for `torch.compile`, whatever the job
  size. For a small job that is most of the runtime. `TORCHDYNAMO_DISABLE=1`
  skips the compile: on 117 molecules ORB optimization went from 75 s to 2.5 s.
- `steps_between_swaps: 1` is the fastest setting; leave it.

## Reproducibility

Conformer embedding runs on the GPU (nvMolKit) when `technical.device` is
CUDA, and on the CPU (morfeus) otherwise. A fixed `conformer_seed` reproduces
the CPU embedding exactly. It does **not** reproduce the GPU embedding: two
identical calls can return different conformer sets. Treat two GPU runs as
independent samples. To compare settings, run replicates, or embed on CPU.
CPU and GPU embedding are different algorithms, so never mix them in one
comparison.

## Starting from SMILES

The initial geometry comes from a force field that ignores spin, and the
MLIP optimizer uses no Hessian, so a SMILES start can occasionally converge
to a saddle point. Start from a reasonable XYZ when the result matters, or
generate an ensemble and keep the lowest energy. Energies are in eV, on
`Structure.energy` in the API.

## Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `unrecognized arguments: --config` | `convert`/`generate` | Run from the config's directory, or use the API |
| `unrecognized arguments: --device` | Global flag after the subcommand | Move it before the subcommand |
| Hugging Face authentication error | UMA without a token, often from a missing config | Pass `--config`; set the token or use ORB |
| CUDA OOM | Batch too large or GPU shared | Lower `max_memory_padding`; pin a free GPU |
| `model_modal` error with SevenNet | Multi-modal checkpoint | Set `model.model_modal` |
| Charged or open-shell XYZ gives wrong energies | Flag defaults 0/1 overrode the config | Pass `--charge`/`--multiplicity` |
| Different conformers on rerun | GPU embedding is non-deterministic | Expected; see Reproducibility |
