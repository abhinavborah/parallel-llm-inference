# parallel-llm-inference

Pipeline-parallel inference for a small language model, split across two machines.

- Stage 0 and Stage 1 each hold part of the model's layers and exchange activations over gRPC.
- A gateway admits requests, orders decode steps, and streams tokens to clients.
- Workers that crash are restarted and rebuild their state by replay; late outputs from old worker incarnations are rejected.
- Output is checked against single-process inference.

Status: early development.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 0.13.0 or newer
- macOS or Windows (Linux should work)

## Model weights

Each model is pinned to a Hugging Face commit, and every file is checked against its sha256.
Files go to `models/<name>/`, which git ignores.

```
uv run python -m pllm.models smollm2-135m
uv run python -m pllm.models qwen3-0.6b
```

## Reference runner and prompt set

The single-process reference runs the same forward pass on CPU in fp32: it tokenises,
prefills and decodes greedily, and prints the tokens and the top logit of each step.

```
uv run python -m pllm.reference smollm2-135m --length 128 --index 0 --new-tokens 16 --threads 4
uv run python -m pllm.reference smollm2-135m --text "The capital of France is" --threads 4
```

`data/prompt_set.json` is the fixed prompt set used by every experiment: 50 passages from
The Adventures of Sherlock Holmes (Project Gutenberg ebook 1661), each cut to 16, 128 and 512
tokens, with token ids stored per pinned tokenizer. Its sha256 is checked on load. To rebuild
it from the book's text file:

```
uv run python -m pllm.prompts path/to/pg1661.txt
```

## Two-stage split in one process

`pllm.split.InProcessSplit.load(model_dir, k)` cuts the model after layer k - 1: stage 0 holds
the embedding and layers 0 to k-1, stage 1 holds the remaining layers, the final norm and the
LM head. Each stage reads only its own tensors from the checkpoint. It has the same
`new_cache` and `forward` calls as the single-process model, so `pllm.reference.generate`
runs on it unchanged, and its logits are bitwise equal to the single-process ones on one host
(fp32, batch 1, the same thread count). With tied embeddings the LM head is the embedding
matrix, so stage 1 loads its own copy. Stage 0 reads the whole embedding into memory; the
memory-mapped lookup for Host B (R3.2) comes with the stage worker.

## Stage worker

`pllm.worker.StageWorker` is one process's share of the model: it loads only its layer
range, logs the weight load time and the process RSS, and calls the step delay once
before each step (zero by default). The first stage reads embedding rows through the
checkpoint's memory map, so only the rows of the tokens it is given enter memory. The
rows are widened to fp32 exactly as a full load would, so the results stay bitwise equal
to the single process on one host. The network service that hosts a worker is a separate
piece of work.

```
uv run python -m pllm.worker smollm2-135m --stage 0
uv run python -m pllm.worker smollm2-135m --stage 1
```

Each command loads the stage, prints the load time and RSS, and exits. The layer range
comes from the split point under `[split]` in the committed `config/hosts.toml` (never a
local copy, so both laptops agree): stage 0 runs layers 0 to k - 1 and stage 1 the rest. `--layers START:END` overrides it for tests and split sweeps.

## X1a: the split against the single process

`pllm.x1a` runs the 50 prompts of one length through the single-process model and through
the two-stage split at the split points 1, n/4, n/2, 3n/4 and n-1. It writes one JSON line
per prompt and split with the exact token match and the maximum logit difference over the
full vocabulary, then a summary line, to a `.jsonl` file next to the run record in `runs/`
(named in the record as `results_file`) and to stdout. It exits with 0 only at 100 percent
token match with finite differences; a non-finite difference is written as `null` with
`non_finite` true, so every line is strict JSON.

```
uv run python -m pllm.x1a smollm2-135m --threads 6 --length 128
```

`--threads` is required and is the same for the reference and the split, because equality
holds only at one thread count. Use the same value on a later run you compare with. The
run record in `runs/` keeps each prompt's reference tokens and top logits with a sha256
that also covers the model, prompt set, length, token count and thread count. The checksum
detects a changed reference on the same host. It differs across hosts, because the top
logits differ in their low bits there, so across hosts compare the stored tokens per prompt
and report agreement. The summary line also gives the inter-op
thread count. The stored logit summary is the top logit per step, not the full logits. `--length` is 16, 128 (the default)
or 512; `--count` limits the run to the first prompts.

## Hosts and ports

Every process reads its addresses from `config/hosts.toml`. The committed file runs all
processes on one laptop. For a LAN run, copy it to `config/hosts.local.toml` (ignored by git),
set each host to that laptop's IP, and set `PLLM_CONFIG` to the copy:

```
# macOS
export PLLM_CONFIG=config/hosts.local.toml
# Windows PowerShell
$env:PLLM_CONFIG = "config/hosts.local.toml"
```

## Run records

Every script writes a JSON record to `runs/` (ignored by git) with the commit, model and
revision, split point, thread count, host, OS, network, round-trip time, available memory,
power state and dates.

## Development

```
uv sync
uv run pytest
uv run ruff check
uv run ruff format
```

`tests/test_model_weights.py` checks the forward pass against the reference library on the
real weights of each model. Each model is skipped until it is downloaded, as in CI. Free
memory needed: about 1.5 GB for SmolLM2-135M and about 5 GB for Qwen3-0.6B.
