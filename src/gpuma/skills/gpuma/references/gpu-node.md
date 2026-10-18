# Working on the GPU node

You run directly on a machine with one or more GPUs that other people may share.
There is no job scheduler: whatever you start runs right away on the GPU you
choose.

## Picking a GPU

- `gpuma doctor --json` lists every GPU with its free memory and returns
  `recommended_device`. Put that value into `technical.device`, or pass
  `--device cuda:N` on the command.
- `nvidia-smi` shows which processes, including other users' jobs, are on each
  GPU. Avoid a GPU on which someone else's large job is running.
- For optimization, a device that does not exist, or CUDA that is not usable,
  is an error (exit code 3), never a silent switch to another GPU or to the
  CPU. Conformer generation alone may fall back to the CPU, with a warning.

## GPU memory

- gpuma fills the GPU on its own by sizing batches; don't split inputs yourself.
- After a CUDA out-of-memory or "Failed to allocate" error, set
  `technical.max_memory_padding` to `0.8` or pick a GPU with more free memory,
  then rerun.

## Hugging Face token (UMA models only)

ORB and SevenNet need no token. For UMA:

1. The user creates a read token at huggingface.co (Settings, Access Tokens)
   and accepts the license on the `facebook/UMA` model page.
2. Store it in a file only the user can read, for example `~/.hf_token` with
   `chmod 600 ~/.hf_token`.
3. Set `model.huggingface_token_file` to that path. Never write the token into
   `config.json` or print it.

`gpuma doctor --json` shows whether a token is found.

## Model downloads and disk space

The first use of a model downloads it into the Hugging Face cache
(`~/.cache/huggingface`, or `HF_HOME` if set), which needs internet access on
the node. If the home directory has a small quota, set `model.model_cache_dir`
to a directory with more space. Later runs load from the cache.

## Long runs

- Shell commands you run time out after a few minutes. Start long runs with
  `nohup ... --json > summary.json 2> run.log &`, note the process ID (`echo $!`),
  and poll with `tail -n 5 run.log`.
- The run has finished once `kill -0 <pid>` fails. An empty `summary.json` then
  means the process died before writing it; `run.log` says why. To stop a run,
  use `kill <pid>`.
- Every process pays a fixed start-up cost: loading the model, and for ORB
  compiling it (about half a minute). For a tiny job, setting
  `TORCHDYNAMO_DISABLE=1` in the environment skips the compile.
- Run one gpuma process per GPU at a time.
