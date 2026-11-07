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
real SmolLM2-135M weights. It is skipped until the model is downloaded, as in CI, and needs
about 1.5 GB of free memory.
