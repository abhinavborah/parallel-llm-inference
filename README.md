# parallel-llm-inference

Pipeline-parallel inference for a small language model, split across two machines.

- Stage 0 and Stage 1 each hold part of the model's layers and exchange activations over gRPC.
- A gateway admits requests, orders decode steps, and streams tokens to clients.
- Workers that crash are restarted and rebuild their state by replay; late outputs from old worker incarnations are rejected.
- Output is checked against single-process inference.

Status: early development.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- macOS or Windows (Linux should work)

## Model weights

Each model is pinned to a Hugging Face commit, and every file is checked against its sha256.
Files go to `models/<name>/`, which git ignores.

```
uv run python -m pllm.models smollm2-135m
uv run python -m pllm.models qwen3-0.6b
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

## Host checks

Host B memory, with every other program closed and the charger in. Also note Task Manager's
"Available" figure (Performance, Memory) at the same time:

```
uv run python -m pllm.hostmem
```

It samples available memory for a minute and proposes the worker budget: the lowest reading
less 0.5 GiB of headroom, rounded down to 0.1 GiB.

Network between the laptops, using the addresses in the hosts file. Stage 0 runs on Windows,
Stage 1 and the gateway on macOS:

```
# Windows, then macOS
uv run python -m pllm.netcheck serve --roles stage0
uv run python -m pllm.netcheck probe --roles stage0
# macOS, then Windows
uv run python -m pllm.netcheck serve --roles stage1 gateway
uv run python -m pllm.netcheck probe --roles stage1 gateway
# multicast: listen on one laptop, send from the other, then swap
uv run python -m pllm.netcheck multicast-listen
uv run python -m pllm.netcheck multicast-send
# Windows, administrator PowerShell: network profiles and inbound rules for the
# Stage 0 port and the multicast port
uv run python -m pllm.netcheck firewall --roles stage0
```

The Windows adapter on the laptops' network must be on the Private profile, with inbound
rules for the Stage 0 port and the multicast port. Reading firewall rules needs an
administrator shell; without one the check reports that and gives no verdict. With several
network adapters, pass `--interface <this laptop's IP>` to the multicast commands.
If the direct network fails, try in order a direct cable, a phone hotspot, then Tailscale, and
change only the hosts file. Every probe, listen and firewall run writes a run record.

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
