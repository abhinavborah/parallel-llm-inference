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
is waiting on (same attempt and step), giving the stage and the reason. A GAP Nack is the
exception: the request replays (see [Replay](#replay)). A stage takes requests only once both route updates have arrived, a few
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

Step ordering (R3.7). Each stage checks every message it takes from a stream, before the
deadline, fencing and cache rules: the step must be above the highest step it has seen for
the same request and attempt. One that is not (a step sent twice, or one overtaken by a
later step) is counted in the servicer's `order_violations` and logged as a warning, and is
then handled as usual, so a resend at a lower `start_pos` still truncates and appends.
Skipped steps are in order, and a new attempt starts afresh, so a replay under a bumped
attempt is not counted. The stage keeps the highest step for the last 4096 (request,
attempt) pairs (about 0.6 MiB at most); a Free does not clear it, so a late step is still
checked. The hop out of Stage 1 is checked by the gateway's result filter (`duplicate` and
`ahead` in `Gateway.discards`). `tests/test_ordering.py` feeds seeded random interleavings,
with steps swapped, sent twice and restarted under a new attempt, through both stages and
checks each stage's count against a separate count of what that stage received.

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
50061. A fault request from the other laptop needs that port reachable; with the Windows
firewall on, as on the college Wi-Fi, it needs an inbound rule (see "Windows setup").

`status` also gives the laptop's free memory (`free_ram_bytes`) and the worker's own over
its process tree: `worker_rss_bytes` now, and `worker_peak_rss_bytes`, the sum of each
process's lifetime peak, which includes loading the model. Windows reports the peak; macOS
and Linux give null for it. Both are null with no worker.

There is no authentication, so use it only on a network you trust, or limit who can reach it
(see "Network"). `status` reports a worker
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

Protocol note: a stage may also send its weight load time with Register, as gRPC metadata
`pllm-load-ns`, an integer of nanoseconds on the stage's own clock. The field is optional,
outside the protobuf schema and backward compatible: a gateway that does not know it ignores
it, and a missing or malformed value is logged as null. The gateway uses it only as a
duration, to split recovery time (see [Recovery time](#recovery-time)).

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
  weights. For KV admission it reads the tensor shapes in `model.safetensors`, not the
  tensors, so the file must be there; without it the gateway does not start.
- A prompt whose prefill would exceed the receive limit from `[limits]` is rejected with
  INVALID_ARGUMENT before anything is sent to Stage 0 (for SmolLM2 at the committed limits,
  more than 512 tokens). A prompt whose request message is itself larger than the limit in
  bytes is refused earlier by gRPC with RESOURCE_EXHAUSTED; it does not reach Stage 0 either.
- The deadline is the client's `timeout_ms`, judged only by the gateway on its own
  monotonic clock: no clock is compared across machines. A waiting request checks its
  deadline at least every 0.05 s and fails with DEADLINE_EXCEEDED once it has passed,
  whatever a stage does, so no request hangs, even with a stage suspended (R5.6). The
  stated tolerance is 0.5 s after the deadline, asserted on loopback; across laptops, add
  the one-way delay of the error to the client. Delays between stages do not add, because
  only the gateway judges the deadline.
- Each step carries `budget_ms`, the time left on the request's deadline when it was sent
  (R5.4). A stage turns it into a deadline on its own monotonic clock when the message
  arrives, and Stage 0 passes on what is left to Stage 1. A step still waiting at a stage
  when that deadline passes is dropped before it runs, with no Nack, and counted in the
  stage's `expired`. A stage's deadline falls after the gateway's by the time the message
  spent in transit, so a step it drops belongs to a request the gateway has already
  failed. A message held in transit, for example in a suspended stage's socket buffer,
  gets its deadline only when read and can still run late; the gateway drops its result.
  A budget of 0 means no deadline.
- A new request finds a stage that is not registered or declared dead: refused with
  UNAVAILABLE. A running request whose stage dies replays instead (see [Replay](#replay)).
  A stream to Stage 0 that fails with no stage declared dead within the heartbeat timeout
  and one more 0.05 s check (an error inside a stage, for example) fails the request with
  UNAVAILABLE.
- The request table (`Gateway.table()`) lists each running request: id, attempt, tokens so
  far, expected step and seconds to its deadline.
- Result filter and exactly-once delivery (R4.7, R4.8). A result is accepted only when its
  request is running and its attempt, its step and its epoch match the request's attempt,
  the step it waits on and the live epoch of Stage 1. Anything else is dropped and counted
  by reason in `Gateway.discards`: `unknown_request` (a late result after the request
  ended), `stale_attempt`, `stale_epoch` (an old Stage 1), `duplicate` (a step already
  delivered), `ahead` (a step not asked for yet) and `stale_nack` (a Nack for no step a
  running request waits on); `Gateway.dropped` is the total. Only an accepted result
  reaches the client, so each step of a request is streamed once and in order.
- When a request ends, for any reason, it leaves the table and a `Free` for it is queued for
  each stage; the stage's heartbeat stream delivers it.
- Each running request holds one of the gateway's server threads. Six are kept for the
  results stream, registration and the heartbeat streams, so at most 14 requests run at
  once.
- Flow control (R5.1, R5.2), set under `[limits]` in `config/hosts.toml`:
  - `admission_queue` (8): a request that finds all 14 running slots taken waits in a
    first-in, first-out queue of this size; one that finds the queue full is refused at
    once with RESOURCE_EXHAUSTED. Time in the queue counts against the deadline, and a
    client that goes away leaves the queue. Each waiting request holds a thread too, so
    the pool is 20 + 8 threads.
  - `credits` (8): steps in flight in the pipeline at once. A request takes a credit
    before each step and gives it back when the result comes. Waiting requests get a
    credit in arrival order, and a request giving one back queues behind them for its
    next step. A running request has at most one step in the pipeline, so credits bind
    only below the 14 running slots. While requests run normally this bounds the steps
    at each stage; a request that fails (a deadline, a refusal, a cancel) gives its
    credit back while its step may still be at a stage, so there the stage inbox is the
    hard bound.
  - `stage_inbox` (16, at least `credits`): messages each stream's inbox, and its queue to
    the next hop, may hold at a stage. A full inbox stops the stage reading, so gRPC flow
    control slows the sender; on a healthy stream nothing is dropped. A stream that ends
    (a route change, a failed next hop) discards what it held, as before.
  - `--admission-queue N|off` and `--credits N` on the gateway (and on `pllm.launch`, which
    passes them on) change the two gateway bounds for one run; `off` removes the admission
    bound, so nothing is refused. `--credits` stays within `stage_inbox`. The metrics
    file's `run` line records the values in use (`admission_queue` is null when off).
    The running slots are not changed.
  - KV admission (R5.3), from `host_b_budget_gib` (0.7), `host_b_runtime_mib` (196) and
    `host_b_margin_mib` (64). Stage 0 runs on Host B, whose worker budget is 0.7 GiB of peak
    RSS for the whole process (R1.7). The gateway subtracts the runtime (the RSS of a fresh
    process after importing torch and the model code, measured on Host B), the margin, and
    the fp32 size of Stage 0's layers at the split in use, read from the shapes in
    `model.safetensors`. The mapped embedding is read with file reads, so it adds nothing.
    What is left is the KV budget. The margin is a proposed figure, not a measurement, for
    the worker memory that runtime plus weights leave out, all inside the 0.7 GiB: a serving
    Stage 0 for SmolLM2 read about 415 MiB idle, about 16 MiB above them (a check on Host B
    on 2026-11-26, not a recorded run), one appended layer (below), the attention scores of
    a 512 token prefill (about 18 MiB) and gRPC message copies. Each request
    reserves (prompt tokens + `max_tokens`) x Stage 0 KV bytes per position (layers x 2 x
    kv heads x head dim x 4) before it may queue. One that does not fit beside the
    reservations already held is refused at once with RESOURCE_EXHAUSTED (its message says
    when it is over the whole budget, so a retry cannot help) and counted in
    `Gateway.kv_rejected` as well as the `rejected` outcome. Queued requests hold their
    reservation, and every way a request ends gives it back. The gateway logs the budget at
    start and writes `kv_budget_bytes` and `kv_bytes_per_position` to the metrics file.
    - SmolLM2 (k = 15): 202.6 MiB of weights leave 254.2 MiB, 23,040 bytes per position,
      11,570 positions. At the committed limits (22 requests of at most 528 positions) 21
      fit, so it binds only when nearly all 22 are that long; with
      longer requests it binds sooner, for example 5 requests of 2,000 positions.
    - Qwen3-0.6B (k = 14): 840.1 MiB of Stage 0 weights alone exceed 0.7 GiB, so no KV
      budget is left. The gateway refuses to start with the numbers; `--no-kv-admission`
      (on the gateway or the launcher) runs it without the check, and the run record then
      has `kv_budget_bytes` null.
    - Limits. It reserves each request's maximum, not measured use (R5.10 tunes it). An
      append copies one layer of one request, so a step briefly holds one more layer (about
      12 MiB at 8,192 positions for SmolLM2). That layer, prefill activations and gRPC
      buffers are worker memory, so they come out of the margin, not the 0.5 GiB headroom
      above the budget (R1.7 keeps that for other programs). The gateway gives a
      reservation back when the request ends, but Stage 0 frees the cache only when the
      `Free` arrives on its heartbeat stream (every 0.5 s), so requests admitted in that gap
      can briefly overlap the freed caches by up to what they cache in that gap.
    - `uv run pytest tests/test_kv_admission.py` checks requests at exactly the budget and
      one byte over, a queued request holding its bytes, and that each way a request ends
      gives its reservation back.
  - `Gateway.queue_peak`, `Gateway.in_flight_peak` and each stage's `inbox_peak` record the
    deepest each has been; `tests/test_backpressure.py` checks them against the bounds
    under overload.
- Many requests run at once (R4.5). They share one stream to Stage 0, so the stages see
  their steps interleaved and Stage 0 works on one request while Stage 1 works on another:
  in `tests/test_concurrency.py` each stage sleeps 0.1 s per step instead of computing,
  so 4 requests of 5 tokens one after another are 20 steps of 0.2 s, while together the
  stages overlap and the work fits in about 21 slots of 0.1 s; the test checks that
  together takes under three quarters of one after another. That shows the stages
  overlapping, not model throughput, which the benchmarks measure.
  Each request's own steps stay strictly in order and give the tokens it gives alone; no
  order between requests is promised, and a shorter request started later can finish
  first. A cancel, a deadline or a refused step ends only its own request. A stage ends
  the shared stream when a message is malformed, when a step fails inside it (including
  running out of memory), or when its next hop fails or moves; then every request on it
  fails with UNAVAILABLE. A request whose prompt and `max_tokens` together exceed the
  model's context length (8192 positions for SmolLM2) is refused with INVALID_ARGUMENT.
  This bounds each request, not Host B's memory: with an even split, one request at the
  cap holds about 180 MiB of Stage 0 cache for SmolLM2 but about 4.4 GiB for Qwen3-0.6B
  (40960 positions), more than Host B has; KV admission (below) bounds the sum.
  One long prefill delays the other requests' next steps at that stage.
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
| `run` | Start | `clock`, `start_ns`, `wall_time`, `commit`, `model`, `revision`, `split`, `network`, `host`, `os`, `python`, `available_memory_bytes`, `power`, `heartbeat_timeout_s`, `step_timeout_s`, `argv` |
| `hop` | Each message the gateway sends or receives | `t_ns`, `hop`, `msg`, `phase`, `rid`, `attempt`, `step`, `bytes`, `payload_bytes`, `positions`, `accepted` |
| `beat` | Each heartbeat of a live epoch (every 0.5 s); one from an epoch declared dead is not logged | `t_ns`, `stage`, `epoch`, `busy_ns`, `steps`, `inbox_depth` |
| `depth` | When the gateway's queue depths have changed, checked every 0.05 s | `t_ns`, `running`, `queued`, `in_flight`, `queue_peak`, `in_flight_peak` (the peaks since the gateway started, which also count a spike between two checks) |
| `request` | Each request's end | `t_ns`, `rid`, `attempt`, `outcome`, `code`, `prompt_tokens`, `max_tokens`, `output_tokens`, `arrived_ns`, `first_token_ns`, `last_token_ns` |
| `bytes` | Run end, per hop, message type and phase | `hop`, `msg`, `phase`, `messages`, `bytes`, `payload_bytes`, `positions`, `source` |
| `detected` | Each failure detection | as in [Failure detection](#failure-detection) |
| `registered` | Each stage registration | as in [Failure detection](#failure-detection) |
| `divergence` | A replayed token differs from the one sent | `t_ns`, `rid`, `attempt`, `step`, `token`, `history_token` |
| `discard` | Each result or Nack the gateway drops; the `run` line then has `discard_lines` true | `t_ns`, `reason`, `source` (`result` or `nack`), `stage` and `epoch` of the sender, `nack_reason`, `rid`, `attempt`, `step` |
| `run_end` | Run end | `t_ns`, `links`, `links_unreported`, `stage0_stage1_complete`, `write_errors` |

- Hops are `gateway_stage0`, `stage0_stage1` and `stage1_gateway`. `phase` is `replay` for a
  replayed step, otherwise `prefill` at step 0 and `decode` after it.
- `bytes` is the protobuf serialized size, without gRPC and HTTP/2 framing. `payload_bytes`
  is the raw tensor and `positions` the tokens a request message carries.
- `outcome` is `end of sequence`, `max_tokens`, `cancelled by the client`, `failed` or
  `rejected`, and `code` is the gRPC status the gateway set. A send time is when the message
  was queued for its stream, not when it left the laptop.
- `busy_ns` and `steps` in a `beat` are the stage's running totals since that process
  started: time spent handling steps, including a step still running, measured on the
  stage's own clock, and the steps handled, not counting those refused with a Nack. They
  are durations, never compared with a gateway time; `t_ns` is when the beat arrived. A
  restarted stage has a new `epoch` and counts from zero. Two streams working at once on
  one stage both count, so `busy_ns` can grow faster than the clock.
- Stage 0 to Stage 1 never passes the gateway, so it has byte totals but no `hop` lines and
  no latency of its own. Stage 0 counts each stream and returns the counts, with no
  timestamp, on that stream's gRPC trailing metadata.
- `stage0_stage1_complete` is false when a stream ended with no valid report, for example
  because Stage 0 died, or was stopped at the same time as the gateway. Its totals are then
  a lower bound. For complete totals stop the gateway first, while the stages are up.
- The gateway runs no model and measures no round-trip time, so the `run` line has no thread
  count and no `rtt_ms`. The launcher's run record has the thread count, and the network
  check has the round-trip time.
- `accepted` is the verdict when a result arrives. A result accepted then can still be
  dropped when its request takes it from the queue, for example after a new Stage 1
  registers. Such a drop is counted in `Gateway.discards` and has a `discard` line.
- A message queued just as its stream ended is counted although it was not sent. A file
  with no `run_end` is from a run that did not stop cleanly. Send SIGINT, or CTRL_BREAK on
  Windows, not a forced kill.

## Report

One command builds every table and plot from the metrics files (`pllm.report`). Reported
numbers come from it, never from the dashboard.

```
uv run python -m pllm.report                  # every runs/*.metrics.jsonl
uv run python -m pllm.report LOG_OR_DIR ...   # given files, or those in each folder
uv run python -m pllm.report --out DIR        # default runs/report
```

It writes `runs.csv` and `index.md` with one row per run, and a folder per run with
`summary.md`, `latency.csv`, `stages.csv`, `bytes.csv`, `recovery.csv`, `latency_cdf.png` and, when the
stages sent beats, `utilisation.png`. The report is built beside the output folder and
then replaces it whole, so a failed build leaves the old one. The command refuses a folder
it did not make, a file, and a folder that holds a log; anything added to a folder it made
is replaced with it. `runs/` is not committed.

Definitions, as in vLLM's serving benchmark, all on the gateway's one clock:

- Completed requests ended at end of sequence or at `max_tokens`; only they give latencies
  and throughput. The others are counted per outcome.
- TTFT is the first token minus arrival at the gateway, so time in the admission queue
  counts. TPOT is (last token minus first token) / (output tokens - 1), for requests with
  at least two tokens. End-to-end is the last token minus arrival.
- p50, p95 and p99 interpolate linearly between the nearest ranks (numpy's default). Each
  table gives n: a p99 of fewer than 100 requests is an interpolation.
- Throughput is the output tokens of completed requests over the window, from the first
  arrival of any request to the last token of a completed one.
- Busy time per stage is read off the beat totals at the two ends of the window, by
  straight-line interpolation between beats, separately per epoch. Since the totals count
  a step still running, the error at each end is at most the time between the two beats
  around it (0.5 s when beats are steady). `max_beat_gap_s` in `stages.csv` is the longest
  stretch of the window with no beat from that stage, for example a late first beat, a
  heartbeat that reconnected or beats that stopped, and `summary.md` warns when it is over
  1 s. Utilisation is busy time over the window; two streams working at once on one stage
  can push it above 1.
- Recoveries: one row per `recovery` line (see [Recovery time](#recovery-time)), times in
  ms. `stale_discarded` is counted here only: the request's results the gateway dropped
  (`hop` lines with `accepted` false) at an attempt from the previous recovery's attempt of
  that request (1 for the first) up to, not including, this recovery's, over the whole run.
  So a late result of a resumed old stage counts even when it arrives after the recovery
  line, and a request that recovers twice counts each result once. Nacks have no `hop` line
  and are not counted (the gateway's `stale_nack` count has them), nor is a result accepted
  on arrival and dropped only when its request took it, after a bump (`Gateway.discards`
  has it).

Plots for each experiment (X2 to X9) are added here by that experiment.

## Dashboard

A live page for the demo (`pllm.dashboard`, standard library only, so nothing for Host B and
no new dependency). It is off unless the gateway is started with `--dashboard`, because it can
kill and freeze workers:

```
uv run python -m pllm.gateway smollm2-135m --dashboard
uv run python -m pllm.launch smollm2-135m --roles gateway stage1 --dashboard
```

Open `http://HOST:PORT/` from a browser. HOST and PORT are the `[dashboard]` table of the
hosts file (default: the gateway's host, port 50080). HOST must be the address the browser
types, not `0.0.0.0` (that is refused at start): the server answers only to that name, and to
`localhost` and `127.0.0.1` when it listens on loopback. The page is one HTML file
with plain JavaScript and uses no address outside the server.

- Stages: each stage alive, dead or unknown with its epoch, the age of its last Beat, its
  worker's state and the free memory of its laptop. Alive means the epoch is live and a Beat
  came within the heartbeat timeout; dead means the failure detector declared the epoch dead.
- System: requests running and queued, steps in flight, tokens per second (over 5 s),
  requests rejected (every request refused or failed before it started, such as a full queue,
  a deadline spent waiting for a slot, or no stage registered), results dropped (all reasons
  of `Gateway.discards`, listed on hover), duplicate tokens and the gateway laptop's free
  memory.
- Send, Send 8 at once and Flood (40 requests, more than the gateway runs and queues): each
  request goes through the gateway's own `Frontend.Generate` on loopback, so deadlines, the
  admission queue and refusals are those of any client, and its tokens stream into a card.
- Kill, Freeze and Unfreeze: sent to the supervisor of that stage's laptop (`[supervisor0]`
  and `[supervisor1]`), which carries them out. A worker that is killed is respawned.

Free memory comes from each supervisor's `/status`; with no supervisor running the page shows
"supervisor unreachable" and the fault buttons answer with an error.

Duplicate tokens are counted on the streams the page relays (a token whose step does not go
above the last of its request); the gateway delivers each step once, so it stays 0. The gateway
counts tokens sent and requests ended by outcome (`Gateway.state()`).

Safety, as for the supervisor's control port: a POST needs `Content-Type: application/json`,
a Host the server answers to and no foreign Origin, no response has a CORS header and OPTIONS
is not answered, so a page on another site cannot press the buttons from a browser. There is no
authentication beyond that: use it only on a network you trust (see "Network"). Open event
streams (16), requests
started from the page (64 at once), each client's backlog and request bodies are bounded. A
viewer that vanishes without closing its connection keeps its place until a write to it
fails, and a viewer that reads slowly loses its oldest events, so a card can miss its first
or last token; reloading the page starts again from the next state.

The tests pin the page, every event it reads and the server's rules, and, when `node` is
installed, run the page's own script against a stand-in for the DOM, fed a real state event
(`tests/dashboard_page_check.js`). No test drives a real browser. Reported numbers come from
the scripts that read the metrics files, never from this page.

## Heartbeat

A serving worker registers, then keeps one `Control.Heartbeat` stream to the gateway
(`pllm.heartbeat`). Every 0.5 s it sends a Beat with its stage, epoch, inbox depth, cached
request count and its busy time and step totals, and it sends each Nack as soon as it is
raised, between beats. The gateway keeps the latest Beat of each stage with its arrival
time on the gateway's own clock (`Gateway.beats()`), writes each one to the metrics file,
and keeps the Nacks (`Gateway.nacks`, the last 1024, and
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
serving; frees still queued for the old process are dropped. A process whose epoch was
declared dead with no replacement yet gets ABORTED at its next Beat or stream, exits with
code 1, and its supervisor starts a new one (see "Failure detection"). A restarted gateway would
refuse every stream the same way; only workers are restarted. Gateway failure is outside
the fault model (R6.16): there is no durable request log, so a gateway that dies loses every
request in flight, and clients must send them again.

Limits:

- A request started between a registration and its route update reaching the other stage
  (a few milliseconds) is refused by fencing and fails at once.
- A Nack or a `Free` sent just as the stream breaks can be lost with it. A lost `Free`
  leaves that request's cache on the stage until the process ends.

## Replay

A declared death bumps the attempt of every request in the gateway's table at once,
including a request that is waiting for a credit with KV on both stages (R6.5). Each
request then gives back its credit, waits until every stage is registered and live, and
runs again from step 0 under its new attempt (R6.6):

- step 0: the prompt, `start_pos` 0, length P;
- step k, for each token already sent: that step's input token, `start_pos` P+k-1, length 1;
- every one of them with `replay` set, through both stages. Then normal decoding goes on.

These are the original messages, so the KV each stage rebuilds is bitwise equal to the
first run on one host (prefill of P+m tokens need not be: `tests/test_kv_replay.py`). A
stage adopts the newer attempt and truncates at position 0. Each replayed result is
compared with the token the client was sent and is not sent again (R6.7). A mismatch is
logged, counted in `Gateway.divergences` and written as a `divergence` line; the request
goes on from the history the client has.

- A GAP Nack (a stage with no cache at the step's position) replays the request too.
- While a request replays, a failed stream or a refused step (a new stage before its route
  update) starts it again under a newer attempt, after a pause: 0.05 s, doubled at each
  refusal up to 1 s, times a random 1 to 1.5, never past the deadline. A new token or a new
  death starts the pause at 0.05 s again. A replay never reuses an attempt, so a late
  result of an older one is always dropped.
- A request waits for the new stage at most until its own deadline, then fails with
  DEADLINE_EXCEEDED. `Gateway.replays` counts replays started.
- A request that lives through 3 stage deaths (`MAX_DEATHS`) fails with UNAVAILABLE at the
  third, instead of replaying again: a request that kills every new stage, for example by
  running a worker out of memory, must not keep the others replaying until its deadline.

Replay is always full, through both stages, even when only Stage 1 died. Partial replay,
where Stage 0 keeps its outbound activations so that only Stage 1 recomputes (R6.15,
experiment X10), was not built: the replay scope decision (#51) chose full replay only.

## Recovery time

Each replay is one recovery (R6.12). When it ends, at the request's first new token or when
the request ends while still replaying, the gateway writes one `recovery` line to the
metrics file. All times are on the gateway's clock except the weight load, which the new
stage reports as a duration (see [Stage registration](#stage-registration)).

| Field | Meaning |
|---|---|
| `rid`, `attempt`, `recovered`, `outcome` | the request, the attempt that recovered, false (with the outcome) if it ended first |
| `stage`, `old_epoch`, `new_epoch`, `path` | the death that bumped the request; null for a replay with no death (a GAP Nack) |
| `detection_ns` | declaration minus the dead epoch's last beat: an upper bound on fault to detection, since the gateway cannot see the fault itself |
| `respawn_ns` | declaration to the new registration, minus the weight load: noticing the exit, starting the process, hashing the weights, registering |
| `weight_load_ns` | the new stage's own load time; null if it sent none (an older worker, or a stage started inside a test) |
| `overlap` | true when the restart ran while detection was still pending (always for `replaced`); `respawn_ns` is then 0 and the parts add up to more than the restart |
| `replay_prefill_ns` | the replayed prompt's send to its result |
| `recomputed_tokens`, `divergences` | replayed steps compared with the history, and those that differed |
| `total_ns` | the dead epoch's last beat to the first new token; null if not recovered |

The four parts do not add up to `total_ns`: the rest is recomputing the tokens sent before
and any pause before a refused replay. Stale outputs discarded are counted by the report from
the metrics file (see [Report](#report)), so the late results of a resumed old stage count
too. If both stages die during one replay, the line names the later death. A request that
fails at a death before replaying, such as at `MAX_DEATHS`, still gets a line, with
`recovered` false.

Both stages killed at once (R6.14): each death is declared and bumps every request, so a
request uses two of its `MAX_DEATHS`, and its replay starts only once both new stages have
registered. A single failure already replays through both stages, so a double failure costs
the slower of the two restarts, which run in parallel (one supervisor per host), plus any
contention: at least as long as either stage's own restart, not twice as long. The gap is
small, so the fault suite checks that bound, not a timing margin. The line names one of the
two deaths, the one the gateway handled last (usually the later declared), so its `total_ns`
starts at that stage's last beat and can miss up to one heartbeat interval of the other.

## Fault suite

Every failure-handling claim of Epic E6 has a test that injects the fault (R6.13). The tests
carry the pytest mark `fault`, so one command runs them all on either laptop:

```
uv run python -m pllm.models smollm2-135m   # once, for the real-weights test
uv run pytest -m fault -rs
```

`-rs` lists every skipped test with its reason, so no test skips unseen. Only
`test_a_real_worker_killed_by_the_fault_injector_is_back_and_registered` can skip: it needs
the downloaded SmolLM2-135M weights and 1.5 GiB of free memory. On Host B (0.7 GiB budget)
that skip is expected: the respawn claim is still tested there with stand-in worker scripts
and with real stage processes of a small random model, and the real-weights respawn runs on
Host A. Every other fault test uses stand-in scripts or that small model, so the suite needs
no downloaded weights. The suite uses only 127.0.0.1.

| Claim | Fault injected | Tests |
|---|---|---|
| R6.1 kill, suspend and resume, per-step sleep | kill and suspend of real process trees; a step delay | `test_faults.py`; `test_supervisor.py::test_kill_suspend_and_resume_are_applied_through_the_fault_injector`, `::test_the_control_port_applies_kill_suspend_and_resume` |
| R6.2 the supervisor respawns, respawn time logged | kill of the worker; a worker that exits or crashes | `test_supervisor.py::test_a_worker_killed_by_the_fault_injector_is_respawned`, `::test_the_respawn_time_is_logged`, `::test_a_crash_loop_backs_off_and_never_gives_up`, `::test_a_killed_worker_comes_back_and_registers_again` |
| R6.3 detection by stream break, heartbeat timeout, step deadline | kill, suspend, a slow step | `test_detection.py::test_a_killed_stage_is_detected_by_its_stream_break`, `::test_a_suspended_stage_is_detected_by_its_heartbeat_timeout`, `::test_a_slow_step_is_recorded_without_declaring_the_stage_and_the_request_completes`, `::test_a_step_held_by_a_suspended_stage_names_it_then_its_timeout_declares_it`; `test_detection_cross.py` |
| R6.4 dead epoch, new registration, route update | a stopped or resumed stage | `test_detection.py::test_a_dead_epoch_waits_for_a_new_registration_and_a_route_update`, `::test_a_stage_resumed_with_no_replacement_is_told_and_exits`, `::test_a_resumed_stage_is_replaced_by_its_supervisor_with_no_manual_step`; `test_detection_cross.py` |
| R6.5 idle requests holding KV are bumped | a stopped stage while one request waits for a credit | `test_replay.py::test_an_idle_request_holding_kv_is_bumped_at_the_declaration_and_both_finish`; `test_replay_cross.py` |
| R6.6 chunked replay through both stages | a stopped stage; a GAP Nack | `test_replay.py::test_the_replay_resends_the_original_messages_with_the_replay_flag`, `::test_a_gap_nack_replays_the_request`; `test_replay_cross.py` |
| R6.7 replay verified, output equals a fault-free run | kill of a real stage process (each stage); a forced mismatch | `test_replay.py::test_a_killed_stage_process_is_replaced_and_the_request_finishes_unchanged`, `::test_a_replayed_token_that_differs_from_the_history_is_logged_and_not_sent`; `test_replay_cross.py` |
| R6.8 late outputs of a resumed stage rejected | suspend past the timeout, resume after the replacement (each stage) | `test_stale_after_suspend.py::test_a_resumed_old_stage_has_every_late_output_rejected` |
| R6.9 attempts, MAX_DEATHS, growing pause, deadline | repeated stage deaths; a refused replay; no replacement | `test_fail_cleanly.py::test_each_declared_death_raises_every_attempt_by_exactly_one`, `::test_a_request_that_lives_through_max_deaths_fails_with_unavailable`, `::test_a_refused_replay_waits_longer_each_time`; `test_replay.py::test_with_no_replacement_the_request_fails_at_its_deadline`; `test_replay_cross.py` |
| R6.10 a slow worker fails cleanly at the deadline, bounds hold | a step delay on each stage; a suspended worker | `test_fail_cleanly.py::test_a_worker_slower_than_the_step_timeout_fails_requests_at_their_deadline_within_bounds`; `test_deadlines.py::test_with_a_suspended_worker_every_request_ends_by_its_deadline` |
| R6.11 a delayed or lost hop | none yet | open: #50, on the hop wrapper of #49 |
| R6.12 recovery time in four parts | kill of a real stage process (each stage); a stopped stage; suspend and resume | `test_recovery.py`; `test_stale_after_suspend.py` (stale outputs counted) |
| R6.14 both stages at once | kill of both real stage processes together | `test_both_stages.py::test_both_stages_killed_together_recover_and_the_request_finishes_unchanged` |

## Failure detection

The gateway watches each stage incarnation by its epoch (`pllm.detection`, read with
`Gateway.detector.events()`). It logs each record as one JSON line and writes it to the
metrics file as a line whose `type` is the record's `event`.

| Path | Fires when | Effect |
|---|---|---|
| `stream_break` | the heartbeat stream of the latest registration ends | epoch declared dead |
| `heartbeat_timeout` | no Beat for `--heartbeat-timeout` seconds (default 2.0, four intervals), checked every 0.05 s | epoch declared dead |
| `replaced` | a new registration arrives before either of the above | epoch declared dead |
| `step_deadline` | a step has no result `--step-timeout` seconds after it was sent (default 10), once per step | record only |

A declared epoch is never revived; the first declaration writes the only record. Requests
are refused with UNAVAILABLE until the stage registers again. The new registration gets a
higher epoch, and the neighbour gets a route update. Requests already running replay once it
has (see [Replay](#replay)).

A step deadline names a suspect, the stage whose last beat is older than half the heartbeat
timeout, and declares nothing. A slow stage still beats, so its supervisor would never send
the registration a declaration waits for, and the gateway cannot tell which stage holds a
late step. The request carries on to its own deadline.

Every reported time is an integer from the metrics file's clock (`time.perf_counter_ns` on
the gateway), in a field ending in `_ns`, so intervals subtract directly and match the
file's other lines. `time` is the wall clock, a label only.
Detection records:

- `detected`: `path`, `stage`, `epoch`, `declared`, `t_ns`, `time`, `last_beat_ns`,
  `since_last_beat_ns`, `rid`, `step`, `detail` (`stage` and `epoch` are the suspect or null
  for a step deadline).
- `registered`: `stage`, `old_epoch`, `new_epoch`, `t_ns`, `time`, `waited_ns` (from the
  declaration), `route_queued_ns` (null if the neighbour had no stream), `load_ns` (the
  weight load time the new stage reported, or null).

The two default timeouts are provisional: the beat gaps behind the 2.0 s value were measured
on one laptop only. A laptop that pauses longer, for example while paging, needs a larger
`--heartbeat-timeout`. The `run` line of the metrics file records the values in use.

Every gRPC server and every channel between the processes (registration, heartbeat, the
gateway's stream to Stage 0 and the stream between the stages) sets keepalive
(`pllm.wire.channel_options`): each end pings after 2 s with no traffic and closes the
connection when the ack is 8 s late. The client's and the dashboard's channels to the
gateway do not, so run them on the gateway's laptop. A link that goes silent without a TCP
reset, such as a Wi-Fi stall or a laptop that left the network, breaks every keepalive
stream on it within 10 s instead of when the operating system gives up. The
stage's heartbeat then opens a new stream, and the gateway's stream to Stage 0 and the
stream between the stages fail instead of hanging. A stall shorter than 8 s breaks
nothing. The gateway refuses a `--heartbeat-timeout` of 8 s or more: a longer stall would
break the heartbeat stream first and declare the stage at once, so the tuned timeout would
never be the one that decides.

Limits:

- A stage that breaks its heartbeat stream is dead to the gateway even if the process is
  fine, so a network blip now costs a respawn and a weight reload.
- A worker suspended and never resumed produces no registration, because its supervisor sees
  a live process. The gateway waits.
- A worker suspended for more than about 8 s has its connections closed by keepalive, since a
  stopped process cannot answer a ping. Outputs it held for a closed stream can then be lost
  rather than delivered late, so a stale-output run (X6) keeps each suspension under 8 s.

## Beat gaps

`pllm.beatgaps` reads a gateway metrics file and prints, for each stage, the gaps between
consecutive beats of one incarnation on the gateway's clock: count, median, p99 and maximum,
for the idle part of the run, the loaded part (inside the request window) and both. It then
prints the heartbeat timeout the run used, the stages the gateway declared dead, a verdict
(pass when nothing was declared and the longest gap is below the timeout; exit code 0 only
then) and a suggested timeout: the longest gap times `--margin` (default 1.5), rounded up to
0.5 s. The suggestion is only a suggestion; the chosen value and its margin are stated with
the measured numbers. Each call writes a run record.

The measurement on two laptops, with no fault injected: 10 minutes idle, then about 10
minutes of load.

```
# macOS (gateway and Stage 1)
uv run python -m pllm.launch smollm2-135m --roles gateway stage1
# Windows (Stage 0)
uv run python -m pllm.launch smollm2-135m --roles stage0
# macOS, after 10 minutes idle; stop it with Ctrl-C after 10 more minutes
uv run python -m pllm.x2 pipeline smollm2-135m --concurrency 4 --repeats 1 --requests 1000 --metrics runs/<gateway>.metrics.jsonl
# macOS, after stopping the load, with both launchers still running
uv run python -m pllm.beatgaps runs/<gateway>.metrics.jsonl
```

Run the recorder before stopping either launcher. Stopping one ends that stage's heartbeat
stream, the gateway declares the stage dead, and the recorder counts that declaration, so a
later call on the same file reports fail. The metrics file is written a line at a time, so
reading it while the gateway runs is safe; the run record of that call holds the measurement.
The recorder needs only the completed requests to place the load window, so stopping the
load early is fine. Then stop both launchers, start the gateway again with the chosen value,
for example `--heartbeat-timeout 3.0` on the macOS launcher, repeat the run in the same
order, and check that the verdict is pass. The value must stay below the 8 s keepalive
timeout.

## X5 and X6: crash recovery and stale outputs

`pllm.x5` kills a stage with requests in flight (R7.12). `pllm.x6` suspends a stage past
the heartbeat timeout, lets a replacement serve, then resumes the old process (R7.13,
R6.8). Each does it to Stage 1 and to Stage 0 in every repeat (`--stages`, default `1 0`),
5 repeats. Both are clients of a running system, run on the gateway's laptop, read the
gateway's metrics file, and send faults to the supervisor of the stage's laptop.

One repeat sends 4 requests of 64 tokens together (`--length` 128 prompts) and injects the
fault once every one has 20 tokens and none has ended.

Reference: before any fault, prompts of the prompt set run one at a time, in order, and the
first 4 that give all 64 tokens are kept. This is the fault-free output of the same running
system. When every address in the hosts file is the runner's laptop (one host, fp32, batch
1), tokens and top logits must match it bit for bit. Across two laptops the tokens must
match, and no bit-exact claim is made.

### Setup on two laptops

Set `PLLM_CONFIG` on both to a hosts file with this session's addresses (see
[Hosts and ports](#hosts-and-ports)). In both files `[supervisor0]` must hold the Windows
laptop's address and `[supervisor1]` the Mac's, or delete both tables to use the stages'
addresses. A copy of `config/hosts.toml` keeps `127.0.0.1` there, and then no fault from
the other laptop arrives. Use the same commit on both laptops.

Each run records the round trip. Measure it before starting the system, since the probe's
echo server uses the stage's port. Stop the echo server afterwards.

```
# Windows, then stop it with Ctrl-C once the probe is done
uv run python -m pllm.netcheck serve --roles stage0
# macOS
uv run python -m pllm.netcheck probe --roles stage0
```

Then start the system. For X6, name each stage's thread count here, since the standby needs
the same one (the launcher writes it to its run record only when it stops):

```
# Windows (Stage 0)
uv run python -m pllm.launch smollm2-135m --roles stage0 --threads <Windows n>
# macOS terminal 1 (gateway and Stage 1)
uv run python -m pllm.launch smollm2-135m --roles gateway stage1 --threads <Mac n>
```

X5 needs nothing more:

```
# macOS terminal 2
uv run python -m pllm.x5 smollm2-135m --metrics runs/<gateway>.metrics.jsonl --rtt-ms <ms>
```

X6 also needs a standby on each laptop whose stage it suspends. A supervisor never restarts
a suspended worker, which is still running and still holds its port, so `pllm.standby`
starts the replacement when the runner asks: a second supervised worker of that stage on
the stage's port plus 100 (50151 or 50152), asked for on the supervisor's port plus 100
(50161 or 50162). It loads no weights while idle. Its `--threads` must be the number given
to that laptop's launcher.

```
# Windows, second terminal
uv run python -m pllm.standby smollm2-135m --stage 0 --threads <Windows n>
# macOS terminal 3
uv run python -m pllm.standby smollm2-135m --stage 1 --threads <Mac n>
# macOS terminal 2
uv run python -m pllm.x6 smollm2-135m --metrics runs/<gateway>.metrics.jsonl --rtt-ms <ms>
```

With the Windows firewall on, the Stage 0 rule under [Windows setup](#windows-setup) must
also allow the replacement's two ports: `-LocalPort 50051,50061,50151,50161`. The standby,
like the supervisor, has no authentication.

**Host B memory.** From the suspend until the repeat is restored, X6 on Stage 0 holds two
Stage 0 workers on the Windows laptop, the suspended one and its replacement. Each is held
to the 0.7 GiB worker budget (R1.7), so the budget for these repeats is 1.4 GiB, plus the
standby process (36 MiB for its Python process, measured on Windows 11 with
`uv run python -c "import psutil, pllm.standby; print(psutil.Process().memory_info().rss)"`).
That is more than the 0.7 GiB budget and more than the lowest available memory in the first
R1.7 measurement, 1.28 GiB, so close other programs first. Each X6 line has the stage
laptop's available memory before the fault and with both workers loaded
(`stage_free_ram_bytes`). X5 and X6 on Stage 1 keep Host B at one worker.

### One laptop

```
uv run python -m pllm.launch smollm2-135m --threads <n>
uv run python -m pllm.x5 smollm2-135m --metrics runs/<gateway>.metrics.jsonl
# for X6, one more terminal each, with the same thread count
uv run python -m pllm.standby smollm2-135m --stage 0 --threads <n>
uv run python -m pllm.standby smollm2-135m --stage 1 --threads <n>
uv run python -m pllm.x6 smollm2-135m --metrics runs/<gateway>.metrics.jsonl
```

### What is reported

Output: one JSON line per reference request, one per repeat, one aggregate per stage with n,
mean, stdev, min and max over its passed repeats, and a summary, in a `.jsonl` file next to
the run record in `runs/` (named in the record as `results_file`) and on stdout. The record
has the network, `--rtt-ms` (required across two laptops), the gateway's commit and its
heartbeat timeout.

- **Recovery times** are the gateway's recovery parts (see [Recovery time](#recovery-time)):
  `detection_ms`, `respawn_ms`, `weight_load_ms`, and per request `replay_prefill_ms` and
  `total_ms` (a repeat's value is their mean, with `total_ms_max`). Detection runs from the
  dead epoch's last beat, so it is an upper bound on fault to detection. The weight load is
  the new worker's own duration; no times of two clocks are subtracted. X5 also gives
  `supervisor_respawn_s`, the supervisor's own time from noticing the exit to the new
  worker listening, weight load included; it overlaps the parts. In X6, `respawn_ms` runs
  to the replacement's registration, so it includes the runner's reaction and the
  standby's start.
- **Stale outputs.** `discards_by_reason` counts every drop in the repeat's lines.
  `late_by_reason` (X6) counts the resumed process's outputs among those written after
  the resume was sent: for Stage 1, results carrying its old epoch; for Stage 0, Stage 1's
  refusals (`stale_nack:BAD_SENDER`) of steps from before the fault. Per request,
  `stale_discarded` is the report's count, which sees dropped results only, not Nacks.
- **Duplicate tokens at the client** are counted per request and summed.
- `held_s` (X6) is the time from sending the suspend to sending the resume, on the
  runner's clock.

A repeat fails, and lists why, when a check fails or cannot be made: the fault did not take
effect (a request ended before it, no declared death, no new epoch, a request that did not
replay from at least token 20), another death or request in the repeat, a missing recovery
line or time, a divergence, a duplicate or missing token, output unlike the reference, or a
dropped result with no `discard` line. X6 also fails a repeat when the death was not found
by the heartbeat timeout, the replacement did not serve, the old process was gone, no late
output arrived, one was accepted (or Stage 1 ran a late step of the old Stage 0), the
stage was suspended for 8 s or more (see "Limits" under
[Failure detection](#failure-detection)), or the system could not be restored. The exit
code is 0 only when every planned repeat of every stage ran and passed. Fewer than 5
repeats, or another load (`--requests`, `--new-tokens`, `--fault-at`), needs `--trial`,
which the record keeps.

**A late output is not certain.** A suspended Stage 0 holds the steps the gateway goes on
sending it, and sends them on when resumed. A suspended Stage 1 loses what it has not yet
read once Stage 0 moves its route to the replacement and closes the old connection, so it
sends a late result only if it was suspended with a step in hand. A repeat in which it held
none shows nothing about rejection and fails. `--extra-attempts N` (default 0) runs such a
repeat again, at most N times per stage, only when that is its one failure. The earlier
attempt stays in the output marked `replaced` and is listed in the summary. Report how many
there were.

The replacement must load and serve before the 8 s are up, 2 s of which pass before the
heartbeat timeout. If repeats fail on that, free memory on the stage's laptop. Detection
also needs the gateway's `--heartbeat-timeout` to stay well below 8 s.

**Restoring.** After each X6 repeat the runner kills the old worker through its supervisor,
which starts a new one that takes over from the replacement, and then stops the
replacement. The suspended worker is resumed on every way out of the runner, errors and
Ctrl-C included. A runner that is killed outright cannot do that: resume the stage by hand
(`curl -X POST -H "Content-Type: application/json" http://<laptop>:50062/resume`, 50061
for Stage 0), kill it the same way with `/kill`, and restart its standby.

`uv run pytest tests/test_x5.py tests/test_x6.py` runs both experiments against real
supervised worker processes of a small random model, with no downloaded weights.

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

## X2: throughput against concurrency

`pllm.x2` serves the same requests at concurrency 1, 2, 4, 8 and 16, either through the
pipeline or in one process round-robin without batching. Both modes run a closed loop: c
requests are outstanding, and a finished one is replaced by the next at once. Each cell has
`--requests` requests (default 32, at least twice the concurrency) over the prompt set's
passages of `--length` tokens (default 128), with `--new-tokens` (default 64), after one
untimed warm-up request. Cells run repeat by repeat (`--repeats`, default 5).

Pipeline mode is a client of a running system. Start it with `pllm.launch`, then pass the
gateway's metrics file (the newest `runs/*.metrics.jsonl`):

```
uv run python -m pllm.launch smollm2-135m
uv run python -m pllm.x2 pipeline smollm2-135m --metrics runs/<gateway>.metrics.jsonl
```

Single mode loads the whole model in its own process, serves at most 14 requests at a time
like the gateway (the rest wait in arrival order), and writes its own metrics file:

```
uv run python -m pllm.x2 single smollm2-135m
```

`--threads` sets its torch threads (default: the physical cores). The run record keeps the
process's lifetime peak RSS after load and at the end, its RSS just after load, and the
largest RSS sampled while serving, so a load peak and the settled size are told apart.
`--budget-mib` names the budget they are read against; the record states it and makes no
claim about a breach.

Throughput, TTFT, TPOT and stage utilisation are computed by `pllm.report` from the metrics
file, so they use the gateway's clock and the definitions given there. In pipeline mode the
runner also counts ordering violations: steps that do not rise per request and attempt on the
gateway's two hops, and client streams whose steps are not 0 to n - 1. The stages' own
ordering counters are not exported, so they are not part of this count.

Output: one JSON line per cell, one line per mode and concurrency with n, mean, stdev, min
and max over the repeats, and a summary, in a `.jsonl` file next to the run record in
`runs/` (named in the record as `results_file`) and on stdout. Each cell line also has a
sha256 of every prompt's tokens, so on one host the two modes can be compared. The exit
code is 0 only with zero ordering violations and no failed request.

## X1b: agreement across the laptops

`pllm.x1b` sends each prompt of an X1a run through the two-laptop pipeline once, one request
at a time, and compares the tokens with each host's single-process reference (the X1a run
records). Across hosts the result is agreement, never exact equality.

First run X1a on each laptop with the same length and token count. Host B's record comes from
issue #17's run; copy its `runs/<utc>-x1a.json` into `runs/` on Host A. Then start the system
(see "Starting the system": Stage 0 on Host B, the gateway and Stage 1 on Host A) and run, on
Host A:

```
uv run python -m pllm.x1b smollm2-135m --metrics runs/<gateway>.metrics.jsonl --reference runs/<host A>-x1a.json runs/<host B>-x1a.json
```

Each record is checked before any request is sent: an x1a record with status ok, its
reference checksum recomputed over its settings, the prompt set and its entries, the same
model, revision, length and token count in every record, and each stored prompt equal to the
prompt set's. The records must come from two different hosts; `--trial` allows records from
one host for a trial run, and the run record then has `trial: true`. The length, token count
and prompts come from the records. `--metrics` is the running gateway's file; it is read to
refuse a gateway serving another model or revision and for the split point, and the first request waits until both stages beat in it (up to `--wait`
seconds, default 120), since a request sent earlier is refused. `--rtt-ms` puts a round-trip
time from `pllm.netcheck` into the run record.

Per prompt and reference:

- `first_divergence`: the first step whose token differs, `null` if none.
- `agreed_tokens`: the steps before it (all if none), out of `compared_tokens`, the tokens
  streamed.
- `max_logit_diff`: the largest difference of the chosen token's logit over the steps up to
  and including the first divergence. The wire and the X1a record carry only that logit, so
  this is not the full-vocabulary difference of X1a.

The gateway stops at end of sequence and the reference does not. A stop on the token the
reference also chose agrees; a stop on another token diverges there. Per reference, the
summary gives sequence agreement (the percent of prompts whose tokens all agree), token
agreement (agreed over compared tokens), the smallest and median first divergence, and the
largest logit difference.

Output: one JSON line per prompt and reference, then one summary per reference, in a `.jsonl`
file next to the run record in `runs/` (named in the record as `results_file`) and on stdout.
The record keeps the command (`argv`), each reference's file, host and checksum, and the
pipeline's tokens and top logits per prompt, so a later comparison needs no rerun. The exit
code is 0 when every request completed; agreement is reported, not asserted.

## X3: bytes per hop

`pllm.x3` measures the serialized bytes per message on each hop for one prompt length. It
sends `--count` prompts (default 5) of `--length` tokens (16, 128 or 512) one at a time with
`--new-tokens` (default 128; decoding stops earlier at end of sequence), then waits for the
gateway to stop and reads its metrics file.

Stage 0 to Stage 1 never passes the gateway: Stage 0 returns its byte counts when its stream
ends, and the gateway writes the totals for the whole run when it stops. So one gateway run
gives one length, and its file must hold this run's requests only. A restarted gateway is not
rejoined by the stages, so start both laptops again for each length. Run the gateway in its
own terminal and stop it first, while the stages are up: the launcher stops Stage 1 before the
gateway when both run under one launcher, which can lose Stage 0's counts.

For each length (16, 128, 512), on a freshly started system:

```
# Windows (Stage 0)
uv run python -m pllm.launch smollm2-135m --roles stage0
# macOS terminal 1 (gateway)
uv run python -m pllm.launch smollm2-135m --roles gateway
# macOS terminal 2 (Stage 1)
uv run python -m pllm.launch smollm2-135m --roles stage1
# macOS terminal 3
uv run python -m pllm.x3 smollm2-135m --length 16 --metrics runs/<gateway>.metrics.jsonl
```

X3 sends its first request once both stages beat in the gateway's file, since a request sent
earlier is refused and its line would fail the run. When X3 prints "stop the gateway now",
press Ctrl-C in terminal 1. X3 waits up to `--wait` seconds (default 600, also the limit for
the stages) for the file's `run_end` line, writes its results, and exits. Then stop
Stage 1 and Stage 0. `--rtt-ms` puts a round-trip time from `pllm.netcheck` into the run record.

Per hop and phase it gives messages, bytes, payload bytes, positions, bytes per position and
bytes per message, and the overhead per message (bytes minus payload over messages): the mean,
and for the gateway's two hops, which have one line per message, also the largest. Bytes are
the protobuf serialized size; gRPC's 5 bytes per message and HTTP/2 framing are not included.

Only Stage 0 to Stage 1 carries activations: the gateway sends token ids to Stage 0 and gets
one token and its logit per step from Stage 1. So the check is on that hop, per phase:

- payload bytes equal positions x hidden size x 4 B exactly (hidden size from the model's
  `config.json`, 576 for smollm2-135m);
- the mean overhead per message is at most 128 B (45 to 49 B measured on loopback with the
  test model; only the shape and position fields grow with the length, by a few bytes);
- its positions equal those the gateway sent to Stage 0, so no step was lost or added.

X3 refuses a metrics file that is for another model or already has requests, an end or an
unreadable line. The run fails, with the reasons in the summary, when the file has no
`run_end` in time, Stage 0's totals are incomplete, a line is unreadable, a request in the
file is not this run's, a request failed, or a request decoded fewer than `--new-tokens`
tokens (it stopped at end of sequence). Output: one JSON line per hop and phase, then a
summary with the measured overhead, each request's decoded token count and the check, in a `.jsonl` file next to the run record
in `runs/` (named in the record as `results_file`) and on stdout. The record keeps the
command (`argv`), the settings, the hidden size and each request's output tokens. The exit
code is 0 only when the check passes.

## X4: latency against injected hop delay

X4 adds a delay d to every message of the two hops that cross the LAN: gateway to Stage 0
(held at the gateway's send) and Stage 0 to Stage 1 (held at Stage 1's receive). Both are on
Host A, so one machine and one clock inject it. The Stage 1 to gateway hop stays on Host A
and is not delayed. The delay is pipelined: each message is released d after it entered the
hop, so messages that enter together leave together. It counts as transit: no budget is
reduced by it, and Stage 1 sets a step's deadline from its release.

Expectation, stated before any result: about 2 x d is added per token (2, 10, 40 and 100 ms
for d = 1, 5, 20 and 50 ms) at concurrency 1, so the slope of per-token latency (TPOT)
against d is 2.0 there. At concurrency 4, queueing at small d hides part of the delay, so a
lower slope is expected; it is reported, not checked. The runner writes this as the first
line of its output. Throughput is about
min(c / T, 1 / slowest stage time) with T = compute + base + 2 x d per token, so X4 runs at
concurrency 1 and 4.

The delay is set while the system runs, so Host B starts once. `--hop-delay-control` makes
the gateway and Stage 1 read d in ms from `runs/hop_delay_ms` every 0.2 s; each writes the
value it applied to `runs/hop_delay_ms.gateway` or `runs/hop_delay_ms.stage1`, and the
gateway writes a `hop_delay` line to its metrics file for each change. Without the flag
nothing reads the file, and the launcher removes it at start, so it starts at 0 ms.

Two laptops, Host B first:

```
uv run python -m pllm.launch smollm2-135m --roles stage0
```

Host A:

```
uv run python -m pllm.launch smollm2-135m --roles gateway stage1 --hop-delay-control
uv run python -m pllm.x4 smollm2-135m --metrics runs/<gateway>.metrics.jsonl
```

On one host, `uv run python -m pllm.launch smollm2-135m --hop-delay-control` and the same
`pllm.x4` command. The flag needs both the gateway and Stage 1 on the laptop.

Before each cell the runner writes d and waits for both acks, so each cell's delay is
confirmed. Defaults: delays 0, 1, 5, 20 and 50 ms (`--delays`), concurrency 1 and 4, 5
repeats, 16 requests per cell, 32 new tokens, prompts of 128 tokens. Delays are interleaved
inside each repeat in an order rotated per repeat, so slow drift of the network spreads over
every delay. Each cell is an X2 pipeline cell with its `delay_ms`. Then one aggregate per
delay and concurrency (throughput, TPOT and TTFT mean and p50, each with n, mean, stdev, min
and max over the repeats), one fit per concurrency (least-squares slope of the per-cell TPOT
mean against d with a 95% confidence interval, whether it holds 2.0, and whether that
concurrency is the checked one) and the X2 summary.

A sleep can wake late, so a message can be held longer than d. The gateway and Stage 1 each
note every message's real hold on their own clock and append it to
`runs/hop_delay_ms.gateway.held.jsonl` or `runs/hop_delay_ms.stage1.held.jsonl`. Each cell
line has, per hop, n, p50, p99 and min of the hold and how many holds were below d (should be
0), and `achieved_delay_ms`, the mean hold over both hops. A second fit per concurrency, of
TPOT mean against `achieved_delay_ms`, sits beside the checked one and is never checked. If
the two slopes differ, late wake-ups account for the difference, not the pipeline.
The fits state; they assert nothing. The file goes to `runs/` next to the run record. At the
end the delay is set back to 0.

Timeouts: `--timeout` (per request, default 300 s) must be at least twice the network time of
one request at the largest delay, and the runner refuses less. The gateway's step timeout
(default 10 s) is far above the 2 x (d + 4.5) ms plus compute a step needs at d = 50.

Not done here: the optional toxiproxy cross-check.

## X7: overload with bounds on and off

`pllm.x7` offers requests to a running gateway faster than it can serve them, to show what
the bounds do (R7.14). It is open loop: requests arrive on a Poisson schedule whether or not
earlier ones have been answered, so a gateway that falls behind builds a queue instead of
slowing the load. The schedule comes from a seed, and every setting is offered the same one.

The setting under test is the gateway's, set when it starts. Start a fresh system for each
one. A restarted gateway does not take back the old stages, so restart both laptops. For
example, on the laptop with the gateway and Stage 1, then on the Windows laptop:

```
uv run python -m pllm.launch smollm2-135m --roles gateway stage1 --admission-queue 8 --credits 8
uv run python -m pllm.launch smollm2-135m --roles stage0
```

On both laptops the hosts file named by `PLLM_CONFIG` needs `[supervisor0]` set to the
Windows laptop's address on the network in use. A copy of `config/hosts.toml` keeps
`127.0.0.1` there: Stage 0's supervisor then listens only on its own laptop, every memory
sample from the other one fails, and the run does not pass. Delete the table to use Stage
0's address, or set it. The supervisor's port (50061) must be reachable.

The settings are the bound on and off, `--admission-queue 8` and `--admission-queue off`,
both with `--credits 8`, then the bound on with `--credits` 1, 2 and 4.

Start the bound-off setting with `--no-kv-admission` as well. KV admission (R5.3) also
reserves for waiting requests, so with it on the queue is still bounded: at the committed
budget, 72 requests of 128 prompt and 32 new tokens. Only running requests hold a cache on
Stage 0, at most the running slots, so turning it off does not let waiting requests use
Host B memory. A bound-off run with any refusal does not pass. At 22 requests of that size
(8 waiting and 14 running), KV admission does not bind in the bound-on settings.

With the first setting running, measure the capacity once and choose a rate well above it.
Use that rate for every setting:

```
uv run python -m pllm.x7 capacity smollm2-135m --metrics runs/<gateway>.metrics.jsonl
uv run python -m pllm.x7 run smollm2-135m --metrics runs/<gateway>.metrics.jsonl --rate <requests per second>
uv run python -m pllm.x7 report runs/<first>-x7.jsonl runs/<second>-x7.jsonl --out runs/x7-report
```

- `run` sends `--requests` requests (default 120) of the prompt set's passages of
  `--length` tokens (default 128) with `--new-tokens` (default 32) and a `--timeout` of
  120 s each. The runner runs on the gateway's laptop, since it reads the metrics file.
- It reads Stage 0's memory every 0.5 s from `[supervisor0]` in the hosts file, so that
  table, if there is one, must name an address of the Windows laptop that this laptop can
  reach. The peak is compared with `--budget-mib` (default 716.8, the 0.7 GiB worker budget
  of R1.7): the worker's lifetime peak, which includes loading the model, or the largest
  sample where there is no lifetime peak (`peak_source` says which; samples can miss a
  spike between two of them). A reading over the budget is over it. Otherwise memory is
  unknown when nothing could be read, when the reading taken after the load failed, or
  when the worker was restarted during the load.
- Each run writes a `.jsonl` file next to its run record in `runs/`: the setting, the
  admission queue's depth over time, and one result with the counts of completed, refused
  (RESOURCE_EXHAUSTED), late (DEADLINE_EXCEEDED) and otherwise failed requests, p50, p95
  and p99 of TTFT, TPOT and end-to-end latency, and the queue and in-flight peaks. The
  percentiles are those of `pllm.report`, over completed requests only: with the bound on,
  read them beside the refused count.
- A run passes (exit code 0) when every offered request has its line, the load was above
  capacity (the offered rate was more than 1.1 times the completion rate, and at least one
  request of this run had to wait or was refused), the admission queue and the steps in
  flight stayed within their bounds, no beat showed a stage inbox over `stage_inbox`, and
  the worker was read as within the budget. Unknown memory does not pass. `failures` in
  the result says why a run did not. A beat adds up the inboxes of a stage's open streams,
  one in a run with no fault, and samples them every 0.5 s.
- `report` writes `table.md`, one row per setting, and `queue_depth.png`. For each run
  with the bound on it says whether the end-to-end p99 is below that of the bound-off run
  with the same credits, with the refused count beside it; a setting with no bound-off run
  is not compared. It refuses runs that were offered different loads.

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

`--heartbeat-timeout SECONDS` passes the gateway's heartbeat timeout through, so it is
accepted only with the gateway role on this laptop. The gateway checks the range (above the
0.5 s beat interval and below the 8 s keepalive timeout); an out-of-range value makes it
exit, and the launcher then stops the rest. See [Beat gaps](#beat-gaps) for how to choose it.

`--no-kv-admission` passes the gateway's flag through (gateway role only): requests are then
admitted without Host B's KV budget (R5.3). The gateway refuses to start without it when the
model's Stage 0 leaves no KV budget, as Qwen3-0.6B does.

Before every run, close other programs and use the same power state. The launcher logs the
available memory and power state at the start, warns on battery, and writes a run record.
For a two-laptop run it also logs a short round trip to each of the other laptop's roles once
their ports answer (see "Run records").

### Windows setup

- Use native Python, not WSL2.
- Keep the charger in and select the best performance power plan or power mode in Settings.
- The launcher is one more process on this laptop: about 39 MiB, plus 5 MiB for the virtual
  environment's `python.exe` trampoline (measured on Windows 11, without `uv` itself). It is
  not part of the worker's 0.7 GiB budget (requirement R1.7) but comes out of the headroom
  kept above it.
- Stop the launcher with Ctrl-C. On Ctrl-Break `uv` itself exits at once with 0xC000013A, so
  the shell sees that code while the launcher finishes the stop behind it.
- On the college Wi-Fi the Windows firewall is on (agreed 2026-11-26), so the Stage 0 and
  supervisor ports need an inbound rule. On the phone hotspot it is off for now, on the
  Public profile (agreed 2026-11-09), and no rule is needed.
  `uv run python -m pllm.netcheck firewall --roles stage0` reports the state and names the
  `python.exe` in use (run it from an administrator PowerShell to get a verdict).

Both ports have no authentication and the college Wi-Fi is shared, so allow only the other
laptop's address. From an administrator PowerShell, with the profile in use, that
`python.exe` and the other laptop's current address:

```
New-NetFirewallRule -DisplayName "pllm stage0" -Direction Inbound -Protocol TCP -LocalPort 50051,50061 -Program "<python.exe path>" -RemoteAddress <other laptop IP> -Profile <profile in use> -Action Allow
```

- A broader Allow rule for the same `python.exe` still lets every device in, so check the
  inbound rules and disable such a rule. Windows may have made one when it first asked about
  Python.
- `uv` can reach its interpreter through two folders (a short version name that is a
  junction to the full one). If the other laptop cannot connect, add the same rule for the
  other path.
- If the other laptop's address changes, update the rule. Its address on the hotspot is
  different, so with the firewall on there, the rule needs that address.
- If it still cannot connect, look for an inbound Block rule for the same `python.exe`,
  which `netcheck firewall` lists. Cancelling the Windows prompt for Python, or leaving a
  network type unticked there, can create one, and a Block rule wins over an Allow rule.

### macOS setup

- Keep the charger in.
- In System Settings, Network, Firewall, allow incoming connections for the Python that `uv`
  runs, or accept the prompt the first time a process listens.

### Network

Both laptops join the college Wi-Fi, the primary network for two-laptop runs; a phone
hotspot is the fallback (changed 2026-11-26). Ethernet and a direct cable are not used.
Addresses can change between sessions: set each host to that session's address for the
laptop and `network` to the network's name (for example `college-wifi`), and with the
firewall on, update the address in the inbound rule. Before any timed run, repeat the
network probe three times in each direction with both laptops plugged in. If the p95
round-trip spread stays high, report it with each timed result or use the hotspot. No level
for "high" is set yet; it is to be agreed on #123.

No port here has authentication, and the college Wi-Fi is shared. The rule under "Windows
setup" limits only the Windows laptop's Stage 0 and supervisor ports. Nothing in this
README limits the ports on the Mac, so any device on that Wi-Fi that can reach it can use
them: the gateway (50050), Stage 1 (50052), its supervisor (50062), which takes kill and
suspend requests, and the dashboard (50080) if it is bound to the LAN address. Keep the
dashboard on 127.0.0.1, as the committed hosts file does, unless the other laptop needs it,
and stop the processes when a run ends.

## Hosts and ports

Every process reads its addresses from `config/hosts.toml`. The committed file runs all
processes on one laptop. For a two-laptop run, copy it to
`config/hosts.local.toml` (ignored by git), set each host to that laptop's IP, and set
`PLLM_CONFIG` to the copy:

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

In the launch record, `rtt_ms` is the median of every timed TCP connect to the other
laptop's roles, five per role once its port answers, and `probe` holds the figures per role. It is a
connect time, not the echo round trip of `pllm.netcheck probe`. It is null in single-host
mode, and a role that never answered before the run stopped is recorded with its error.
The launch record of a session holds the network name and this round trip for every run in
that session, so the records of scripts run during it leave `rtt_ms` null unless given one.

## Session checklist

At the start of every two-laptop session, on the college Wi-Fi or the phone hotspot fallback,
in order. Steps 2 to 4 stay set, so check them once per laptop.

1. Both laptops on the charger; Windows on the best performance power mode.
2. Windows: Power Options, "Wireless Adapter Settings", "Power Saving Mode" set to Maximum Performance.
3. Windows: Device Manager, Wi-Fi adapter, Power Management, untick "Allow the computer to turn off this device to save power".
4. macOS has no Wi-Fi power saving switch: in Battery, Options, Low Power Mode off and "Prevent automatic sleeping on power adapter when the display is off" on.
5. Hotspot fallback only: phone on its charger, hotspot kept on (idle auto-off disabled), updates and sync paused on both laptops.
6. Read both addresses: `ipconfig getifaddr en0` on macOS, `ipconfig` on Windows (Wi-Fi IPv4 address).
7. On both laptops set each host and `network` (`college-wifi` or the hotspot name) in `config/hosts.local.toml`, and `PLLM_CONFIG` (see "Hosts and ports").
8. College Wi-Fi or any network other than our own hotspot: Windows firewall on with the TCP 50051 and 50061 rules under "Windows setup" (limiting them to the Mac's address is optional); check with `uv run python -m pllm.netcheck firewall --roles stage0` as administrator.
9. Run the two-way TCP probe under "Host checks"; both directions pass before any run.
10. Start both launchers (see "Starting the system"); each launch record holds the network name and the short round trip.

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

With the Windows firewall off on a profile (as on the phone hotspot for now, agreed
2026-11-09), the check reports every port as allowed there. With it on (as on the college
Wi-Fi), the check reads the inbound rules for the profile in use, which needs an
administrator shell (without one it gives no
verdict), and prints a rule to add for that profile and the interpreter. The printed rule
has no address limit, so on the college Wi-Fi use the form under "Windows setup" instead.
The check also does not read a rule's remote addresses: it reports a port as allowed
whether or not the rule is limited. With several network adapters, pass
`--interface <this laptop's IP>` to the multicast commands.

With the Windows firewall on, the Stage 0 rule covers TCP only, so a multicast listen on
the Windows laptop needs its own rule for UDP 50059; without it, hearing nothing says
nothing about the network. Add it from an administrator PowerShell for the test and remove
it afterwards, or run the multicast test on the hotspot with the firewall off:

```
New-NetFirewallRule -DisplayName "pllm multicast test" -Direction Inbound -Protocol UDP -LocalPort 50059 -Program "<python.exe path>" -RemoteAddress <other laptop IP> -Profile <profile in use> -Action Allow
Remove-NetFirewallRule -DisplayName "pllm multicast test"
```

On a new session, check the addresses again (see "Network" above). Every
probe, listen and firewall run writes a run record.

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
