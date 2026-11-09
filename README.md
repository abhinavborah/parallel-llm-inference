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
to the single process on one host. `--serve` hosts the worker on the network, see
[Chain over gRPC](#chain-over-grpc).

```
uv run python -m pllm.worker smollm2-135m --stage 0
uv run python -m pllm.worker smollm2-135m --stage 1
```

Each command loads the stage, prints the load time and RSS, and exits. The layer range
comes from the split point under `[split]` in the committed `config/hosts.toml` (never a
local copy, so both laptops agree): stage 0 runs layers 0 to k - 1 and stage 1 the rest. `--layers START:END` overrides it for tests and split sweeps.

## Chain over gRPC

`pllm.stage_service` serves `Stage.Steps` for a worker. Stage 0 takes token ids, runs its
layers and sends the hidden states on to stage 1 as a `StepRequest`; the last stage runs
the LM head, takes the greedy token and its logit and pushes a `StepResult` to the
gateway's `Results.Push`. Steps are handled in arrival order, and each request id has its
own KV cache in `pllm.kvstore.KVStore`. Fencing by epoch is a separate piece of work: the
epoch fields are carried but not checked.

KV cache rules (R3.6). A message whose `start_pos` equals the cached length appends. A lower
`start_pos` truncates the cache to it and then appends, which is how the gateway replays
from an earlier message; truncating at the end of an earlier message and appending is
bitwise equal to a fresh run. A higher `start_pos`, or a request the stage has not seen
that does not start at 0, is refused: the message is skipped and not passed on, a `GAP`
Nack is put on the servicer's `nacks` queue, and the stream and the other requests on it
carry on. `free` releases a request's cache on a stage, does nothing for an unknown request
and returns the cached count to 0 when every request is freed (`cached_requests` is the
count a Beat reports). The heartbeat stream that sends Nacks to the gateway and delivers
Free messages is a separate piece of work, so for now nothing leaves the process and the
tests call `free` directly. A request costs per position, per stage, `layers x 2 x kv
heads x head dim x 4` bytes: for SmolLM2-135M 23,040 bytes, so a 528-position request holds
11.6 MiB on each stage and 8 such requests 93 MiB (steady state; while a step
appends, a layer's cache is briefly about twice its size). Requests that never receive a Free
stay cached until the process ends, and the `nacks` queue is unbounded until the heartbeat
stream drains it.

Activations travel as raw little-endian fp32 bytes with a dtype name and a shape
(`pllm.wire`), never as JSON, and are bitwise equal on arrival. The receive limit of every
server and channel comes from `[limits]` in the committed `config/hosts.toml`:
`max_batch x max_prompt_tokens x hidden size x 4` bytes plus a 64 KiB margin, instead of
gRPC's 4 MiB default.

```
uv run python -m pllm.worker smollm2-135m --stage 1 --serve --threads 6
uv run python -m pllm.worker smollm2-135m --stage 0 --serve --threads 6
```

Each process loads its layers and listens on its port from the hosts file (stage 1 pushes
results to the gateway's address, stage 0 forwards to stage 1's). `--threads` sets the
torch thread count of the process, which every step uses (without the option, torch's
default for the machine); equality runs fix it. On one host, with the same thread
count, the tokens and top logits through the chain equal the in-process split's
(`tests/test_chain.py`; the SmolLM2 cases are skipped when the weights are not downloaded,
the others use a small random model). A stage that fails a step drops that request's
cache, and a failed next hop ends the stream with UNAVAILABLE even if the sender holds it
open.

Once it is listening, a worker registers with the gateway at the gateway address in the
hosts file (R3.3) and keeps the epoch it is given. It hashes `model.safetensors` first, which
takes a second or so, and tries again every second while the gateway is not reachable. A
gateway that rejects the registration (another model, weights or layer range) ends the worker
with exit code 1.

## Supervisor

`pllm.supervisor` runs one laptop's worker as a child process and restarts it whenever it
exits, for any reason, with no manual step (R6.2). A worker that dies while loading, or
within 5 s of it starting to listen, is a crash: it is restarted after a growing pause of 0,
0.5, 1, 2 and then 5 s, and the supervisor never gives up, so a worker that the gateway keeps
rejecting is restarted every 5 s until someone fixes the configuration. A worker that is
killed through the control port is not counted as a crash. A worker that has not started
listening after 3 minutes is killed and started again. A suspended worker is alive, so it is
not restarted; noticing a stalled worker is the gateway's job (R6.3).

```
uv run python -m pllm.supervisor smollm2-135m --stage 1 --threads 6
```

It writes JSON lines to `runs/supervisor_stage1.jsonl` (`--log` changes the path): `spawned`,
`ready`, `exited`, `backoff`, `fault` and `stopped`, each with `time` (this laptop's clock)
and `mono` (monotonic seconds). A `ready` line after a restart carries `respawn_seconds`:
from the moment the exit was noticed to the moment the worker accepted a connection on its
port, which includes loading its layers. Registration follows and is logged by the worker.
Ctrl-C, SIGTERM or Ctrl-Break stops the supervisor and its worker. If the supervisor is
force killed instead, its worker checks once a second that the supervisor is still running
and exits when it is not, so it does not keep the port.

Measured on Windows 11 with SmolLM2-135M, stage 1, 2 threads: a worker killed by the fault
injector was listening again after 4.2 s (run
`uv run pytest tests/test_supervisor.py -k real_worker --basetemp=<dir>` and read
`events.jsonl` under it). A Python process that has imported the supervisor is 36 MiB
(`uv run python -c "import psutil, pllm.supervisor; print(psutil.Process().memory_info().rss)"`).

Kill, suspend and resume requests go to the supervisor's control port and are applied to
the worker's process tree through `pllm.faults` (SIGKILL on macOS and Linux, TerminateProcess
on Windows). A POST must send `Content-Type: application/json`, which a web page cannot do
to another host without a preflight that the server never answers:

```
curl -X POST -H "Content-Type: application/json" http://<laptop>:50062/kill
curl -X POST -H "Content-Type: application/json" http://<laptop>:50062/suspend
curl -X POST -H "Content-Type: application/json" http://<laptop>:50062/resume
curl http://<laptop>:50062/status
```

The address comes from `[supervisor0]` and `[supervisor1]` in the hosts file, and the
supervisor listens on that host; without the tables it is the stage's host on port 50061 or
50062. The examples use 50062, stage 1 on the Mac; for stage 0 on the Windows laptop use
50061. A fault request from the other laptop needs that port open: on the Windows laptop,
allow it inbound on the Public profile, the same profile as the worker port (see "Starting
the system").

There is no authentication, so keep it on the lab network. `status` reports a worker
as suspended from the moment a suspend is applied until it is resumed or replaced; a suspend
that lands as the worker exits can briefly mislabel its replacement.

## Stage registration

`pllm.control` holds the gateway side of `Control.Register`; the gateway serves one model per
run (`expected_model`, with an optional split point for sweeps). A stage sends its index,
layer range, model id, the sha256 of the weights it loaded and its address. The gateway
rejects a model it does not serve, a hash that differs from the pinned one, or a layer
range that does not match the split point, and otherwise returns a new epoch.
Epochs come from one counter in the gateway, so a stage that registers again always gets
a higher epoch. `pllm.control.register` is the stage side.

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

## Starting the system

`pllm.launch` starts this laptop's processes the same way on macOS and Windows: the gateway
through `pllm.gateway` and each stage through `pllm.supervisor`. With no `--roles`, every
process starts on this laptop (single-host mode, for tests and X1a):

```
uv run python -m pllm.launch smollm2-135m
```

For a run across the laptops, set `PLLM_CONFIG` to your hosts file on both (see "Hosts and
ports"), then start each laptop's share. Order does not matter, since a stage retries until
the gateway is up:

```
# macOS (Stage 1 and the gateway)
uv run python -m pllm.launch smollm2-135m --roles gateway stage1
# Windows (Stage 0)
uv run python -m pllm.launch smollm2-135m --roles stage0
```

Send a prompt once both stages have registered and received each other's route update, a
moment later. Until then the client gets UNAVAILABLE and can send the prompt again.

```
uv run python -m pllm.client "The capital of France is" --max-tokens 16 --timeout 30
```

The launcher refuses a role whose address in the hosts file is not on this laptop, and a model
with any pinned file missing. By default each stage gets this laptop's physical cores divided by its stages, so on
one host each stage gets half; `--threads` sets it. Ctrl-C stops everything. If a process
exits by itself, the launcher stops the rest and exits with code 1 (a worker that dies is
not such a case: its supervisor restarts it). Closing the terminal also stops everything on
macOS. Force killing the launcher leaves its processes running.

Before every run, close other programs and use the same power state. The launcher logs the
available memory and power state at the start, warns on battery, and writes a run record.

### Windows setup

- Use native Python, not WSL2.
- Keep the charger in and select the best performance power plan or power mode in Settings.
- The launcher is one more process on this laptop: about 39 MiB, plus 5 MiB for the virtual
  environment's `python.exe` trampoline (measured on Windows 11, without `uv` itself), a small
  part of the roughly 2.5 GB this laptop has for the worker.
- Allow the Stage 0 port and the supervisor port on the Public profile, from an administrator
  PowerShell:

```
New-NetFirewallRule -DisplayName "pllm stage 0" -Direction Inbound -Protocol TCP -LocalPort 50051 -Profile Public -Action Allow
New-NetFirewallRule -DisplayName "pllm supervisor 0" -Direction Inbound -Protocol TCP -LocalPort 50061 -Profile Public -Action Allow
```

Both ports have no authentication, so remove the rules (`Remove-NetFirewallRule -DisplayName
"pllm stage 0"`, and the same for the other) when the laptop joins a network you do not trust.

### macOS setup

- Keep the charger in.
- In System Settings, Network, Firewall, allow incoming connections for the Python that `uv`
  runs, or accept the prompt the first time a process listens.

### If the network fails

Try in order a direct cable, a phone hotspot, then Tailscale. Only the hosts file changes: set
each host to that network's address for the laptop and `network` to the network's name.

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
