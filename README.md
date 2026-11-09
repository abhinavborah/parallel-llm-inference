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
own KV cache in `pllm.kvstore.KVStore`.

Fencing (R3.11). Before anything is cached, each message is checked in this order, and a
refused one is not cached or forwarded; its Nack goes to the gateway on the heartbeat stream:

1. `target_epoch` is not this stage's epoch: `WRONG_EPOCH` (aimed at a dead incarnation).
2. `sender_epoch` is not the live upstream epoch: `BAD_SENDER`. Stage 0 accepts only the
   gateway (epoch 0); Stage 1 accepts Stage 0's current epoch, which the gateway sends it as
   a route update when its heartbeat stream opens and each time Stage 0 registers. This
   fences a resumed old Stage 0.
3. an attempt older than the newest one that has used the request's cache: `STALE_ATTEMPT`.
   A newer attempt is adopted (a replay starts at position 0, which truncates).

The gateway fails a running request at once with UNAVAILABLE when a Nack names the step it
is waiting on (same attempt and step), giving the stage and the reason; replaying it instead comes with the recovery
stories. A stage takes requests only once both route updates have arrived, a few
milliseconds after the second stage registers: a request sent before that is refused with
`WRONG_EPOCH` (or `BAD_SENDER`) and fails at once. When the route update then arrives,
Stage 0 ends the stream that carried the stale epoch, so the request can be sent again.
Only a Nack for the step a request is waiting on fails it; one for another step (a late
message from a resumed old stage) leaves it running. A stage that has not registered yet
has epoch 0 and accepts any message whose epochs are both 0.

KV cache rules (R3.6). A message whose `start_pos` equals the cached length appends. A lower
`start_pos` truncates the cache to it and then appends, which is how the gateway replays
from an earlier message; truncating at the end of an earlier message and appending is
bitwise equal to a fresh run. A higher `start_pos`, or a request the stage has not seen
that does not start at 0, is refused: the message is skipped and not passed on, a `GAP`
Nack is put on the servicer's `nacks` queue, and the stream and the other requests on it
carry on. `free` releases a request's cache on a stage, does nothing for an unknown request
and returns the cached count to 0 when every request is freed (`cached_requests` is the
count a Beat reports). A Free names an attempt; the stage remembers the last 4096 frees
(about 0.8 MiB) and refuses a later message of that attempt or an older one with a
`STALE_ATTEMPT` Nack, so a step still in flight when its Free arrives cannot cache the
request again (after 4096 later frees the stage forgets one, and a step that late would
cache it again). A newer attempt is accepted, and a Free for an older attempt leaves the
cache a newer attempt holds. `STALE_ATTEMPT` therefore also marks a late step of a request
that has ended, which needs no action; recovery acts only on one for a running request.
The heartbeat stream (under "Heartbeat") sends the Nacks to the gateway and delivers the
frees. A request costs per position, per stage, `layers x 2 x kv
heads x head dim x 4` bytes: for SmolLM2-135M 23,040 bytes, so a 528-position request holds
11.6 MiB on each stage and 8 such requests 93 MiB (steady state; while a step
appends, a layer's cache is briefly about twice its size).

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
allow it inbound on the Private profile, the same profile as the worker port, from an
administrator PowerShell:

```
New-NetFirewallRule -DisplayName "pllm supervisor 0" -Direction Inbound -Protocol TCP -LocalPort 50061 -Profile Private -Action Allow
```

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

## Gateway

`pllm.gateway` serves `Control.Register`, `Control.Heartbeat`, `Frontend.Generate` and
`Results.Push` on the
gateway port from the hosts file. Start it first, then each stage worker with `--serve`
(they register with it), then send a prompt with the client:

```
uv run python -m pllm.gateway smollm2-135m
uv run python -m pllm.worker smollm2-135m --stage 1 --serve
uv run python -m pllm.worker smollm2-135m --stage 0 --serve
uv run python -m pllm.client "The capital of France is" --max-tokens 16 --timeout 30
```

For each request the gateway tokenises the prompt (raw text, no special tokens), assigns a
request id and attempt 1, and sends the whole prompt to Stage 0 as one message at
`start_pos` 0. Each result from Stage 1 is streamed to the client with its step, text and
top logit, and its token goes back to Stage 0 as the next step, one position further on.
A request ends at the model's end-of-sequence token or after `max_tokens` tokens. Decoding
is greedy, and on one host, in fp32, at batch 1 and one thread count the tokens and top
logits equal the single-process reference (tested with small random models).

- The gateway runs on the Mac and loads only the tokenizer and `config.json`, not the
  weights.
- A prompt whose prefill would exceed the receive limit from `[limits]` is rejected with
  INVALID_ARGUMENT before anything is sent to Stage 0 (for SmolLM2 at the committed limits,
  more than 512 tokens). A prompt whose request message is itself larger than the limit in
  bytes is refused earlier by gRPC with RESOURCE_EXHAUSTED; it does not reach Stage 0 either.
- The deadline is the client's `timeout_ms`, judged only by the gateway on its own
  monotonic clock. A request past it fails with DEADLINE_EXCEEDED. Carrying the remaining
  time on each hop comes with the deadline story.
- A stage that is not registered, or a stream to Stage 0 that fails (Stage 0 or Stage 1
  down), fails the request with UNAVAILABLE. Recovery and replay come later.
- The request table (`Gateway.table()`) lists each running request: id, attempt, tokens so
  far, expected step and seconds to its deadline.
- When a request ends, for any reason, it leaves the table and a `Free` for it is queued for
  each stage; the stage's heartbeat stream delivers it.
- Each running request holds one of the gateway's 20 server threads. Six are kept for the
  results stream, registration and the heartbeat streams, so at most 14 requests run at
  once and a further one is refused with RESOURCE_EXHAUSTED; many in flight is the next
  story. Requests share one stream to
  Stage 0, and a step that fails at a stage ends that stream, so the other requests on it
  fail with UNAVAILABLE too.
- Text is decoded per token, holding back a token that ends inside a character until the
  next one completes it (up to four tokens), so a `Token` can carry empty text.

## Metrics

The gateway always writes one JSON-lines file per run, to `runs/<utc>-gateway.metrics.jsonl`,
or to the new file named by `--metrics PATH`. One gateway process is one run. A program that
hosts `Gateway` itself ends a run with `close()` and `recorder.finish()`, then can assign a
new `recorder`. Prompt text is never logged.

Every timestamp is a `*_ns` integer from one `time.perf_counter_ns` at the gateway, so no
time from another machine is subtracted. `wall_time` on the first line is a label only.

| `type` | When | Fields |
|---|---|---|
| `run` | Start | `clock`, `start_ns`, `wall_time`, `commit`, `model`, `split`, `network` |
| `hop` | Each message the gateway sends or receives | `t_ns`, `hop`, `msg`, `phase`, `rid`, `attempt`, `step`, `bytes`, `payload_bytes`, `positions`, `accepted` |
| `request` | Each request's end | `t_ns`, `rid`, `attempt`, `outcome`, `code`, `prompt_tokens`, `max_tokens`, `output_tokens`, `arrived_ns`, `first_token_ns`, `last_token_ns` |
| `bytes` | Run end, per hop, message type and phase | `hop`, `msg`, `phase`, `messages`, `bytes`, `payload_bytes`, `positions`, `source` |
| `run_end` | Run end | `t_ns`, `links`, `links_unreported`, `stage0_stage1_complete`, `write_errors` |

- Hops are `gateway_stage0`, `stage0_stage1` and `stage1_gateway`. `phase` is `replay` for a
  replayed step, otherwise `prefill` at step 0 and `decode` after it.
- `bytes` is the protobuf serialized size, without gRPC and HTTP/2 framing. `payload_bytes`
  is the raw tensor and `positions` the tokens a request message carries.
- `outcome` is `end of sequence`, `max_tokens`, `cancelled by the client`, `failed` or
  `rejected`, and `code` is the gRPC status the gateway set. A send time is when the message
  was queued for its stream, not when it left the laptop.
- Stage 0 to Stage 1 never passes the gateway, so it has byte totals but no `hop` lines and
  no latency of its own. Stage 0 counts each stream and returns the counts, with no
  timestamp, on that stream's gRPC trailing metadata.
- `stage0_stage1_complete` is false when a stream ended with no valid report, for example
  because Stage 0 died, or was stopped at the same time as the gateway. Its totals are then
  a lower bound. For complete totals stop the gateway first, while the stages are up.
- A message queued just as its stream ended is counted although it was not sent. A file
  with no `run_end` is from a run that did not stop cleanly. Send SIGINT, or CTRL_BREAK on
  Windows, not a forced kill.

## Heartbeat

A serving worker registers, then keeps one `Control.Heartbeat` stream to the gateway
(`pllm.heartbeat`). Every 0.5 s it sends a Beat with its stage, epoch, inbox depth and
cached request count, and it sends each Nack as soon as it is raised, between beats. The
gateway keeps the latest Beat of each stage with its arrival time on the gateway's own
clock (`Gateway.beats()`), and the Nacks (`Gateway.nacks`, the last 1024, and
`nack_count`). It sends back:

- the `Free` messages queued for that stage;
- to Stage 0, a route update with Stage 1's address and epoch, when Stage 0's stream opens
  and each time Stage 1 registers. Stage 0 then sends to that address with that target
  epoch without restarting, on a new channel even at the same address (an old channel
  that failed while Stage 1 was down would wait out gRPC's reconnect backoff, up to 120
  s; Stage 0's channels do not share connections, so the new one starts fresh). A stream
  still open on the old route is ended with UNAVAILABLE, and the gateway's next request
  opens a new one.
- to Stage 1, a route update naming stage 0 with its address and epoch, when Stage 1's
  stream opens and each time Stage 0 registers. Stage 1 takes that epoch as the only
  sender epoch it accepts (see "Fencing").

If the gateway cannot be reached, the stage tries again every second, and its Nacks wait
in an unbounded queue. When a stage registers again, the gateway ends the old process's
stream with FAILED_PRECONDITION and that process stops sending heartbeats but keeps
serving; frees still queued for the old process are dropped. A restarted gateway would
refuse every stream the same way; only workers are restarted.

Limits:

- Declaring a stage dead from missed beats, and replaying its requests, come with the
  recovery stories. A request started between a registration and its route update reaching
  the other stage (a few milliseconds) is refused by fencing and fails at once.
- A Nack or a `Free` sent just as the stream breaks can be lost with it. A lost `Free`
  leaves that request's cache on the stage until the process ends.

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

## Host checks

Host B memory, with every other program closed and the charger in. Also note Task Manager's
"Available" figure (Performance, Memory) at the same time, and write it in the R1.7 row of the
BRD with the date:

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
# Windows: network profile and firewall state of each adapter, and a verdict per port
uv run python -m pllm.netcheck firewall --roles stage0
```

For now the Windows laptop stays on the Public profile with the Windows firewall off
(agreed 2026-11-09), so no inbound rule is needed and the check reports every port as
allowed on Public. If the firewall is turned on again, the check reads the inbound rules
for the profile in use, which needs an administrator shell (without one it gives no
verdict), and prints a rule to add for that profile and the interpreter. With several
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
real weights of each model. Each model is skipped until it is downloaded, as in CI. Free
memory needed: about 1.5 GB for SmolLM2-135M and about 5 GB for Qwen3-0.6B.
