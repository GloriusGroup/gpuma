# Choosing config values

`gpuma config --schema` is the authoritative list of keys, types, defaults and
allowed values; this file says which values to pick. Unknown keys are kept but
ignored. Misspelled values of `model.model_type`, `model.model_name`,
`optimization.batch_optimizer`, `technical.memory_scales_with` and
`technical.device` are errors (exit code 3), reported by `--dry-run` and before
every run.

## Models

| `model.model_type` | `model.model_name` | Use for | Notes |
| --- | --- | --- | --- |
| `orb` | `orb_v3_direct_omol` | Organic and molecular systems, the default choice | No token. `orb_v3_conservative_omol` gives energy-conserving forces at higher cost. |
| `fairchem` | `uma-s-1p2` | When the user asks for UMA | Needs a Hugging Face token, see gpu-node.md. First load takes minutes. `uma-m-1p1` is larger and slower. |
| `sevennet` | `7net-omni` | When the user asks for SevenNet | Multi-modal: also set `model.model_modal`, e.g. `omol25_high` for molecules. |

The `*_omat` and `*_mpa` ORB models are trained on inorganic crystals; don't use
them for molecules.

## Optimization

| Key | Default | Guidance |
| --- | --- | --- |
| `optimization.batch_optimizer` | `fire` | Use `lbfgs`. Never `bfgs`, which is slower than `fire`. |
| `optimization.batch_optimization_mode` | `batch` | Keep `batch`. `sequential` optimizes one structure at a time and is only for debugging. |
| `optimization.force_convergence_criterion` | `0.05` eV/Å | Tighten (e.g. `0.01`) only when the geometry feeds something force-sensitive, such as frequencies. |
| `optimization.energy_convergence_criterion` | `null` | Leave unset; the force criterion is the reliable one. |
| `optimization.charge` / `optimization.multiplicity` | `0` / `1` | Used for XYZ input. SMILES input takes its charge from the SMILES. |

## Conformers

| Key | Default | Guidance |
| --- | --- | --- |
| `conformer_generation.max_num_conformers` | `20` | Raise it (50 to 200) for flexible molecules with many rotatable bonds. `--conformers` overrides it per command. |
| `conformer_generation.conformer_seed` | `42` | Reproduces CPU embedding only; GPU embedding varies between runs. |

## Dispersion

`model.d3_correction` adds a D3 dispersion correction. Leave it `false` for UMA
and the ORB OMol models: their ωB97M-V training data already includes
dispersion, so D3 would count it twice. Consider it only for SevenNet models
trained on PBE data.

## Memory and batching

| Key | Default | Guidance |
| --- | --- | --- |
| `technical.device` | `cuda` if a GPU is visible | Use the `recommended_device` from `gpuma doctor --json`. |
| `technical.max_memory_padding` | `0.95` | Lower it to `0.8` after a CUDA out-of-memory error or on a GPU other jobs use. |
| `technical.memory_scales_with` | `auto` | Leave on `auto`; it picks the right metric per model. |
| `technical.steps_between_swaps` | `1` | Fastest setting; leave it. |

gpuma sizes its batches automatically. Don't split inputs into chunks yourself.
