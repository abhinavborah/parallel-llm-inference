# Pipeline-parallel LLM inference with fault recovery: final report

Status: draft skeleton, 2026-11-27, at commit df83e19. Every section is drafted from what is already true on `main`. Result cells hold only results posted on closed issues; every other cell says "pending (#N)" with the issue that will produce it. The members rewrite their own sections (author line under each heading) and fill the results from the runs (#62).

Rules for every number in this report:

- Numbers come from the scripts and their run records (`pllm.report`, the experiment runners in `pllm.x1a` to `pllm.x7`), never from the dashboard.
- Bit-exact equality is claimed only on one host, fp32, batch 1. Across the two laptops the result is agreement: agreement rate, first divergence position, largest logit difference.
- Latencies are measured at the gateway on one clock (`time.perf_counter_ns`). No time from one machine is subtracted from a time on the other.
- Host B (Stage 0) has a worker budget of 0.7 GiB peak RSS for the whole worker process.
- Two-laptop runs use the college Wi-Fi, with a phone hotspot as the fallback. The measured round-trip spread is stated with the results (section 8.3).
- A figure that was a check and not a recorded run is labelled as such.

## Contents

1. Introduction
2. Limitations
3. System description and architecture
4. Protocol
5. Guarantees and failure model
6. What was built
7. Contribution
8. Evaluation setup
9. Results
10. Evidence for Spec R1 to Spec R16
11. Evidence per claim and experiment
12. How to run
13. Conclusion

## 1. Introduction

Author: joint.

Large language models are often too big or too slow for one device. Pipeline model parallelism splits the layers across machines and passes activations between them. This project runs a small Llama-family model, SmolLM2-135M-Instruct, in CPU fp32 across two worker processes on two laptops, with a gateway that admits requests and drives greedy decoding.

The model would fit on one machine. It is split to show the distributed mechanisms, and the smaller laptop (Host B) forces a memory-aware split. The mechanisms are ordered per-request state, bounded queues, timeouts, restart of the failed worker with prompt replay, and rejection of stale outputs.

The central claims are:

- the pipeline's tokens equal single-process inference: bit-exact on one host, measured agreement across the two hosts;
- every token reaches the client exactly once, although computation is at-least-once;
- stale outputs from a dead or paused worker incarnation are fenced out by request, attempt, step and epoch;
- backpressure keeps queues and memory bounded.

## 2. Limitations

Author: joint.

These are stated before the results, because they shape how the results read.

- **Single-request latency is higher than single process.** Each token crosses two network hops (gateway to Stage 0, Stage 0 to Stage 1) and the hop back to the gateway. The pipeline is expected to gain throughput with concurrency, up to about 2x over the pipeline at concurrency 1, and that gain is not promised.
- **The gateway is a single point of failure.** It holds the authoritative token history, the request table and the epochs. There is no durable request log, so gateway failure is outside the fault model. Only workers are restarted.
- **Computation is at-least-once; delivery is exactly-once.** A replay recomputes the prompt and every token already sent. The gateway compares each recomputed token with the history and does not send it again, so the client sees each step once.
- **Agreement, not equality, across architectures.** Host A is macOS arm64 and Host B is Windows x86-64. The low bits of the logits differ across them, so two-laptop runs report agreement, never bit-exact equality.
- **A batched single process may win.** The comparison point that could beat the pipeline, a batched single process on Host A, is not measured: micro-batching is not built (#40 open), and that upper bound waits on it (R7.19). X2 compares the pipeline with single-process round-robin serving without batching.

Also:

- **Host B memory is tight.** The worker budget is 0.7 GiB peak RSS. The clean measurement of available memory gave a minimum of 1.28 GiB (2026-11-03, 60 samples); a reading with other programs open gave a minimum of 0.73 GiB (2026-11-07). Memory pressure and paging on Host B were reported during the first runs on the college Wi-Fi (#60, #124), so runs need other programs closed.
- **The network is shared Wi-Fi.** The first college Wi-Fi probes on 2026-11-26 had p95 round trips from 16 to 120 ms, with both laptops on battery (section 8.3). Later probes the same day were higher (p95 up to 179 ms, not yet on a closed issue). #123 requires a plugged-in rerun before timed runs; Host A's side was run (not yet on a closed issue) and Host B's is pending (#62).
- **Failure detection is tuned for one network.** The heartbeat timeout default of 2.0 s was set from beat gaps on one laptop. Runs on the college Wi-Fi use `--heartbeat-timeout 5` until #124 is settled. A stage whose heartbeat stream breaks is treated as dead even if the process is fine, so a network blip costs a respawn and a weight reload.
- **No authentication.** No port has authentication. The Windows firewall rule limits the Stage 0 and supervisor ports to the other laptop's address; the ports on the Mac are open to the Wi-Fi.
- **One model presented.** All presented runs use SmolLM2-135M (#61). Qwen3-0.6B is pinned and its forward pass is tested, but its Stage 0 weights alone (840.1 MiB, computed from the checkpoint's tensor shapes) exceed the Host B budget.
- **Not built:** micro-batching (#40), partial replay (X10, decided against in #51), gateway durability, discovery and pairing, the toxiproxy cross-check for X4. Both stages failing at once (R6.14) is open in #52.

## 3. System description and architecture

Author: Member A for the gateway, metrics and launcher; Member B for the model, the stage worker, the supervisor and the fault injector.

### 3.1 Processes and placement

| Process | Laptop | Holds |
|---|---|---|
| Gateway (`pllm.gateway`) | Host A, macOS arm64, 24 GB | Tokenizer and `config.json` only; request table, epochs, admission, decode loop, result filter, metrics |
| Stage 1 (`pllm.worker --stage 1 --serve`) | Host A | Layers k to n-1, final norm, LM head, greedy argmax |
| Stage 0 (`pllm.worker --stage 0 --serve`) | Host B, Windows 11 x86-64 | Embedding (memory mapped) and layers 0 to k-1 |
| Supervisor (`pllm.supervisor`), one per laptop | Both | Starts its worker, restarts it after any exit, applies kill, suspend and resume |

```
client --Generate--> gateway --Steps--> Stage 0 --Steps--> Stage 1 --Push--> gateway --Token--> client
                       ^  ^                |                   |
                       |  +---Heartbeat----+                   |
                       +--------------Heartbeat----------------+
           (Beats and Nacks up; route updates and Frees down)
```

SmolLM2-135M has 30 layers and a hidden size of 576. The committed split point is k = 15 (`[split]` in `config/hosts.toml`). This is the even split, not the capacity-aware one: the function that chooses k from each host's free memory and speed, and the X9 sweep runner, are on `main` since PR #140; the sweep has not run yet (#64).

### 3.2 Model and reference

The forward pass (`pllm.model`) is written in PyTorch: RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, explicit position-indexed KV tensors, and QK-norm for Qwen3. It loads safetensors weights directly. Hugging Face transformers is used only in tests, as the oracle. `pllm.reference` is the single-process runner on the same forward code. `pllm.split.InProcessSplit` cuts the model in one process; its logits are bitwise equal to the single process on one host.

Each model is pinned to a Hugging Face commit and every file is checked against its sha256 (`pllm.models`). The prompt set (`data/prompt_set.json`) holds 50 passages of The Adventures of Sherlock Holmes (Project Gutenberg ebook 1661), each cut to 16, 128 and 512 tokens, checksummed on load.

### 3.3 Stage worker

A worker loads only its layer range. Stage 0 reads embedding rows through the checkpoint's memory map, so only the rows of the tokens it sees enter memory, and the result stays bitwise equal to a full load. Each request id has its own KV cache (`pllm.kvstore`). For SmolLM2-135M a request costs 23,040 bytes per position per stage (layers x 2 x kv heads x head dim x 4, computed).

### 3.4 Gateway

The gateway tokenises the prompt, assigns a request id and attempt 1, sends the prompt to Stage 0 as one message at `start_pos` 0, and then sends one token per step. It owns the authoritative history, applies admission control, judges every deadline on its own clock and filters results before anything reaches the client.

### 3.5 Supervision, configuration and launch

The supervisor restarts its worker after any exit with no manual step, with a growing pause (0, 0.5, 1, 2, then 5 s) for crash loops, and never gives up. Every address comes from `config/hosts.toml` (or the copy named by `PLLM_CONFIG`). `pllm.launch` starts a laptop's share the same way on macOS and Windows and writes a run record.

### 3.6 Measurement

The gateway writes one JSON-lines metrics file per run. `pllm.report` builds every table and plot from those files. Each experiment runner writes its own results file next to a run record holding the commit, model, revision, split, thread count, host, OS, network, round-trip time, available memory and power state.

## 4. Protocol

Author: joint.

gRPC, package `pllm.v1`, schema version 3 (`pllm.proto.SCHEMA_VERSION`). The schema is in `src/pllm/proto/protocol.proto`; `tests/test_protocol.py` checks that the services match the agreed RPCs and that a schema change bumps the version.

| Service | RPC | Direction | Carries |
|---|---|---|---|
| Control | `Register(RegisterReq) returns (RegisterResp)` | stage to gateway | stage index, layer range, model id, weights sha256, address; returns the epoch |
| Control | `Heartbeat(stream StageMsg) returns (stream ControlMsg)` | both ways | up: `Beat` (stage, epoch, inbox depth, cached requests, busy ns, steps) and `Nack`; down: `RouteUpdate` and `Free` |
| Stage | `Steps(stream StepRequest) returns (Empty)` | gateway to Stage 0, Stage 0 to Stage 1 | rid, attempt, step, `start_pos`, length, target and sender epochs, replay flag, token ids or activations, `budget_ms` |
| Results | `Push(stream StepResult) returns (Empty)` | Stage 1 to gateway | rid, attempt, step, epoch, token, top logit |
| Frontend | `Generate(GenerateReq) returns (stream Token)` | client to gateway | prompt, `max_tokens`, `timeout_ms` (relative); a stream of tokens with step, text and top logit |

- Activations are a `Tensor`: dtype name, shape and raw little-endian fp32 bytes, never JSON, bitwise equal on arrival.
- The receive limit of every server and channel is `max_batch x max_prompt_tokens x hidden size x 4` bytes plus 64 KiB, from `[limits]` in the hosts file, in place of gRPC's 4 MiB default. A prompt above it is refused with INVALID_ARGUMENT before Stage 0.
- Deadlines: each step carries `budget_ms`, the time left when it was sent. A stage turns it into a deadline on its own clock. The absolute wall-clock deadline field (9) was retired, so no clock is compared across machines.
- Nack reasons: `WRONG_EPOCH`, `BAD_SENDER`, `STALE_ATTEMPT`, `GAP`. Every Nack travels on the heartbeat stream, so the gateway learns of a refusal at once.
- Optional gRPC metadata `pllm-load-ns` on Register carries the new stage's weight load time, used only as a duration to split recovery time. It is outside the protobuf schema and backward compatible.
- Every channel among the gateway and the stages (registration, heartbeat, the gateway's stream to Stage 0, the stream between the stages) sets keepalive: a ping after 2 s without traffic, and the connection closes when the ack is 8 s late. The client's and the dashboard's channels to the gateway do not set keepalive.

## 5. Guarantees and failure model

Author: Member A for the gateway side (5.4 to 5.8); Member B for the stage side (5.1 to 5.3).

### 5.1 Per-request ordering

Each request is strictly sequential; requests interleave freely and no order between them is promised. Each stage checks that the step rises per (request, attempt) and counts a violation. The gateway's result filter checks the hop out of Stage 1. Tests: `tests/test_ordering.py` (seeded random interleavings through both stages), `tests/test_concurrency.py`.

### 5.2 KV state

A message whose `start_pos` equals the cached length appends; a lower one truncates and then appends; a higher one is refused with a GAP Nack, which makes the gateway replay. Truncating at a message boundary and appending is bitwise equal to a fresh run. A Free releases a request's cache on both stages. Tests: `tests/test_kvstore.py`, `tests/test_kv_replay.py`, `tests/test_cross_worker.py`, `tests/test_heartbeat.py`.

### 5.3 Stage fencing

Checked in this order before anything is cached or forwarded: a target epoch that is not this stage's (`WRONG_EPOCH`); a sender epoch that is not the live upstream epoch (`BAD_SENDER`, which fences a resumed old Stage 0); an attempt older than the newest one that used the cache (`STALE_ATTEMPT`). A newer attempt is adopted. Tests: `tests/test_fencing.py`, `tests/test_cross_worker.py`.

### 5.4 Result filter and exactly-once delivery

A result is accepted only when its request is running and its attempt, step and epoch match the request's attempt, the step it waits on and the live epoch of Stage 1. Anything else is dropped and counted by reason: `unknown_request`, `stale_attempt`, `stale_epoch`, `duplicate`, `ahead`, `stale_nack`. Only accepted results reach the client. Tests: `tests/test_results.py`, `tests/test_cross_result_filter.py`.

### 5.5 Failure detection

| Path | Fires when | Effect |
|---|---|---|
| `stream_break` | the heartbeat stream of the latest registration ends | epoch declared dead |
| `heartbeat_timeout` | no Beat for the heartbeat timeout (default 2.0 s), checked every 0.05 s | epoch declared dead |
| `replaced` | a new registration arrives first | epoch declared dead |
| `step_deadline` | a step has no result after the step timeout (default 10 s) | record only |

A declared epoch is never revived. The new registration gets a higher epoch from one counter, and the neighbour gets a route update. Tests: `tests/test_detection.py`, `tests/test_detection_cross.py`.

### 5.6 Replay

A declared death bumps the attempt of every request in the table, including idle requests that hold KV. Each request then resends its original messages under the new attempt, with the replay flag, through both stages: the prompt at `start_pos` 0, then one single-token message per token already sent. Replayed tokens are compared with the history and not sent again; a mismatch is logged as a divergence. A refused replay waits 0.05 s, doubled up to 1 s, times a random 1 to 1.5, never past the deadline. A request that lives through 3 stage deaths fails with UNAVAILABLE (`MAX_DEATHS`). Tests: `tests/test_replay.py`, `tests/test_replay_cross.py`, `tests/test_fail_cleanly.py`.

### 5.7 Deadlines

Only the gateway judges a request's deadline, on its own monotonic clock. A waiting request checks it at least every 0.05 s, so no request hangs, even with a stage suspended; the stated tolerance is 0.5 s after the deadline on loopback. Tests: `tests/test_deadlines.py`.

### 5.8 Backpressure

At most 14 running requests (the constant `MAX_RUNNING` in `gateway.py`). From `[limits]` in the hosts file: an admission queue of 8 (a full queue refuses at once with RESOURCE_EXHAUSTED); 8 credits for steps in flight; a stage inbox of 16 messages per stream, so a full inbox slows the sender through gRPC flow control. KV admission reserves each request's maximum KV on Stage 0 against Host B's KV budget: 0.7 GiB less the runtime (196 MiB, a check on Host B, not a recorded run), a margin (64 MiB, proposed, not measured) and Stage 0's weights (202.6 MiB for SmolLM2-135M at k = 15, computed), leaving 254.2 MiB. Tests: `tests/test_backpressure.py`, `tests/test_kv_admission.py`.

### 5.9 Fault model

In the model: a worker that crashes or is killed, a worker that is suspended and later resumed, a slow worker, a hop message that is lost or delayed, a link that goes silent (keepalive breaks it within 10 s). Out of the model: gateway failure, Byzantine workers, security beyond limiting who can reach the ports. Every failure-handling claim has a test that injects the fault, marked `fault` (section 11).

## 6. What was built

Author: joint.

| Component | Modules | Requirements |
|---|---|---|
| Model, weights, reference, prompt set | `model`, `models`, `reference`, `split`, `prompts` | R2.1 to R2.10, R1.9 |
| Stage worker and KV store | `worker`, `stage_service`, `kvstore`, `wire`, `ordering`, `heartbeat` | R3.1 to R3.13 |
| Registration and epochs | `control` | R3.3, R4.4 |
| Gateway: admission, decode loop, result filter, replay | `gateway`, `client` | R4.1 to R4.8, R5.1 to R5.6, R6.4 to R6.12 |
| Failure detection | `detection`, `beatgaps` | R6.3 |
| Supervisor, fault injector, standby | `supervisor`, `faults`, `standby` | R6.1, R6.2, R7.13 |
| Configuration, run records, launcher, host checks | `config`, `runrecord`, `launch`, `hostmem`, `netcheck` | R1.6 to R1.8, R1.10, R8.2, R8.10 |
| Metrics and report | `metrics`, `report` | R7.1 to R7.6 |
| Capacity-aware split and process memory | `capacity`, `procmem` | R8.6, R8.7 |
| Experiment runners | `x1a`, `x1b`, `x2`, `x3`, `x4`, `x5`, `x6`, `x7`, `x9`, `x9probe`, `x9monitor` | R7.8 to R7.14, R7.16 |
| Dashboard (stretch) | `dashboard` | R9.2 to R9.5 |

## 7. Contribution

Author: joint. Member A is GitHub `abhinavborah` (Host A, macOS). Member B is GitHub `murthymetta2002` (Host B, Windows).

The table puts the owner named in `docs/brd.md` beside the git authors of each module, so it can be checked against history. Commit counts are at commit df83e19, without merge commits, and grow as work lands. Regenerate them with:

```
git log --no-merges --format=%an -- src/pllm/<module>.py | sort | uniq -c
```

A row marked "differs" is one where the BRD owner and the git authors disagree. Both members settle each one, by changing the BRD owner or this table, before submission.

| Module | BRD owner | Commits by A | Commits by B | Match |
|---|---|---|---|---|
| `model.py` | B (R2.1) | 1 | 2 | matches |
| `reference.py` | B (R2.4) | 1 | 2 | matches |
| `split.py` | B (R2.5) | 3 | 3 | shared |
| `prompts.py` | B (R2.6) | 3 | 0 | differs |
| `models.py` | B (R1.9) | 0 | 1 | matches |
| `worker.py` | B (R3.1, R3.2) | 4 | 7 | matches |
| `stage_service.py` | B (R3.4, R3.6, R3.11) | 4 | 10 | matches |
| `kvstore.py` | B (R3.5) | 1 | 4 | matches |
| `wire.py` | B (R3.4) | 1 | 1 | shared |
| `ordering.py` | B (R3.7) | 0 | 1 | matches |
| `heartbeat.py` | B (R3.10) | 1 | 2 | matches |
| `control.py` | A (R4.4), B (R3.3) | 3 | 2 | shared |
| `gateway.py` | A (R4.1 to R4.8, R6.4 to R6.12) | 17 | 15 | shared, differs in balance |
| `client.py` | A (R4.1) | 0 | 1 | differs |
| `detection.py` | A (R6.3) | 3 | 0 | matches |
| `beatgaps.py` | A (R6.3) | 1 | 0 | matches |
| `supervisor.py` | B (R6.2) | 1 | 4 | matches |
| `faults.py` | B (R6.1) | 4 | 3 | shared |
| `standby.py` | A (R7.13) | 0 | 1 | differs |
| `config.py` | A (R1.6) | 4 | 5 | shared |
| `runrecord.py` | A (R1.10) | 1 | 3 | differs |
| `launch.py` | A and B (R8.2) | 7 | 3 | shared |
| `hostmem.py` | B (R1.7) | 0 | 1 | matches |
| `netcheck.py` | A and B (R1.8) | 2 | 3 | shared |
| `metrics.py` | A (R7.1, R7.2) | 2 | 2 | shared |
| `report.py` | A (R7.3) | 1 | 1 | shared |
| `x1a.py` | B (R7.8) | 0 | 2 | matches |
| `x1b.py` | B (R7.8) | 2 | 0 | differs |
| `x2.py` | A (R7.9) | 1 | 0 | matches |
| `x3.py` | B (R7.10) | 2 | 0 | differs |
| `x4.py` | A (R7.11) | 2 | 0 | matches |
| `x5.py` | A (R7.12) | 0 | 1 | differs |
| `x6.py` | A (R7.13) | 0 | 1 | differs |
| `x7.py` | A (R7.14) | 0 | 3 | differs |
| `x9.py` | B (R7.16) | 0 | 1 | matches |
| `x9probe.py` | B (R7.16) | 0 | 1 | matches |
| `x9monitor.py` | B (R8.7) | 0 | 1 | matches |
| `capacity.py` | A (R8.6) | 0 | 1 | differs |
| `procmem.py` | B (R8.7) | 0 | 1 | matches |
| `dashboard.py` | unassigned (R9.2) | 0 | 1 | stretch |
| `proto/protocol.proto` | A and B (R1.4) | 1 | 3 | shared |

Cross-tests, each written by the member who did not build the component (R3.14, R4.9):

| Test file | Component tested | Written by |
|---|---|---|
| `tests/test_cross_worker.py` | stage worker KV truncate and fencing (R3.14) | A |
| `tests/test_cross_result_filter.py` | gateway result filter (R4.9, split agreed in #31) | A |
| `tests/test_replay_cross.py` | gateway replay (R4.9) | B (first commit), later changes by B and A |
| `tests/test_detection_cross.py` | gateway failure detection | B (first commit), later changes by A |

Experiments and report sections: each member runs the experiments they own in the BRD, makes those plots and writes those results subsections (section 9). Owners per the BRD: A owns X2, X4, X5, X6, X7; B owns X1a, X1b, X3, X8, X9.

## 8. Evaluation setup

Author: joint.

### 8.1 Hosts

| | Host A | Host B |
|---|---|---|
| OS and architecture | macOS, arm64 | Windows 11, x86-64 |
| Memory | 24 GB | 5.85 GiB total; available minimum 1.28 GiB, median 1.55 GiB with other programs closed (2026-11-03, 60 samples, `pllm.hostmem`) |
| Cores | 12 physical (#17) | 6 physical (#17) |
| Roles | gateway, Stage 1 | Stage 0 |
| Worker budget | none set | 0.7 GiB peak RSS for the whole worker process |

### 8.2 Model and settings

SmolLM2-135M-Instruct at revision 12fd25f, CPU, fp32, greedy decoding, split k = 15 unless a sweep says otherwise. Thread counts are fixed per process and recorded in each run record; equality runs use the same count for the reference and the pipeline. The PyTorch version is the same on both hosts (one `uv.lock`).

### 8.3 Network

Both laptops on the college Wi-Fi; a phone hotspot is the fallback. No Ethernet or cable. Before any timed run the network probe is repeated three times in each direction with both laptops plugged in.

| Network | Date | Round trip | Notes |
|---|---|---|---|
| Phone hotspot | 2026-11-07 | median 8.67 to 9.26 ms, p95 10.35 to 13.04 ms | `pllm.netcheck probe`, run records on #7 |
| College Wi-Fi | 2026-11-26 | median 8 to 13 ms, p95 16 to 120 ms | First probes of the day: `pllm.netcheck probe`, three runs each way, both laptops on battery, cause of the spread unknown (BRD K5; no run records named). The amendment on #123 gave the p95 low end as 24 ms; PR #132 corrected it to 16 ms, since 24 ms is the lowest Mac to Windows run and one Windows to Mac run read 16.01 ms. Later probes the same day were higher (p95 up to 179 ms, not yet on a closed issue) |
| College Wi-Fi, plugged in | Host A side: 2026-11-26; Host B side: pending (#62) | Host A side: run, not yet on a closed issue; Host B side: pending (#62) | required before timed runs by #123 |

Each timed result states the round-trip spread of its session.

### 8.4 Run discipline

Other programs closed, both laptops on the charger, the same power state for every run, Windows on the best performance power mode, available memory logged at the start of each run. The README "Session checklist" lists the steps.

## 9. Results

Each subsection gives the expectation stated before the run, the command, the plot, and the result. Plots are committed under `docs/figures/<experiment>_<plot>.png` once the runs exist; none is committed yet.

### 9.1 X1a: the split against the single process (Member B)

Expectation: 100 percent exact token match on each host at every split, batch 1, fp32, one thread count.

Result: on both hosts, 250 of 250 exact token match (50 prompts of 128 tokens x splits 1, 7, 15, 22, 29, 16 new tokens) and a maximum logit difference of 0.0 over the full vocabulary (#17). Host A: commit cb385cd, 6 threads, 2026-11-09. Host B: commit 86b3436, 3 threads, 2026-11-26, other programs not all closed.

### 9.2 X1b: agreement across the laptops (Member B)

Expectation: agreement reported against each host's single-process reference; no equality claim.

Result: pending (#55). One request across the laptops (#54) agreed on 16 of 16 tokens with the Host A reference, compared by decoded text since the gateway does not log token ids; this is a first check, not X1b.

### 9.3 X2: throughput against concurrency (Member A)

Expectation: throughput rises with concurrency up to about 2x the pipeline at concurrency 1; single-request latency is higher than single process; zero ordering violations.

Plots: `docs/figures/x2_latency_cdf.png` and `docs/figures/x2_utilisation.png` (from `pllm.report`); a throughput against concurrency plot is not yet made by a script.

Result: pending (#58).

### 9.4 X3: bytes per hop (Member B)

Expectation: on Stage 0 to Stage 1, payload bytes equal positions x 576 x 4 B exactly, and the mean overhead per message is at most 128 B.

Result: pending (#55).

### 9.5 X4: latency against injected hop delay (Member A)

Expectation: about 2 x d added per token at concurrency 1, so the slope of TPOT against d is 2.0; a lower slope at concurrency 4, reported and not checked.

Plot: `docs/figures/x4_tpot_vs_delay.png`, not yet made by a script; the runner writes the fits.

Result: pending (#49).

### 9.6 X5: crash recovery (Member A)

Expectation: after a kill of each stage at token 20 of 64 with 4 requests in flight, every request completes with output equal to the fault-free reference, zero duplicate tokens, and each recovery raises the attempt by one.

Result: pending (#60). Recovery time in four parts: pending (#60).

### 9.7 X6: stale outputs after a suspend (Member A)

Expectation: every late output of the resumed old process is rejected, and the client sees zero duplicate tokens. Run on Stage 1 only, as decided by Member A on #60 (2026-11-27): on Stage 0 it holds two Stage 0 workers on Host B, about 1.4 GiB. Unconfirmed until Member B agrees.

Result: pending (#60).

### 9.8 X7: overload with bounds on and off (Member A)

Expectation: with bounds on, queue depth stays capped and end-to-end p99 is bounded, with the refused count beside it; with bounds off, the queue grows.

Plot: `docs/figures/x7_queue_depth.png` (from `pllm.x7 report`).

Result: pending (#59).

### 9.9 X8: micro-batching (Member B)

Optional clause. Not built; #40 open.

### 9.10 X9: capacity-aware split (Member B)

Expectation: on SmolLM2-135M an even split fits Host B, so the splits separate on speed and busy time, not memory; Host B peak RSS within budget with no paging.

Result: pending (#64, runner on `main` since PR #140).

## 10. Evidence for Spec R1 to Spec R16

Author: joint. Test paths are under `tests/`; every test named here carries the claim it proves in its name. "Run" cites a posted result or the issue that will produce one.

| Clause | Summary | Requirements | Tests | Run or plot |
|---|---|---|---|---|
| Spec R1 | Small model, two worker processes | R1.11, R1.13, R2.1, R3.1, R8.1 | `test_worker.py::test_a_worker_holds_and_caches_only_its_layers`, `test_split.py::test_each_stage_reads_and_holds_only_its_parts` | #54: one request across both laptops |
| Spec R2 | CPUs or GPUs | R2.10, R3.15, R8.1, R8.2 | CI on macOS, Windows and Linux; `test_launch.py` | X1a on macOS arm64 and Windows x86-64 CPUs (#17) |
| Spec R3 | Split layers between workers | R2.5, R2.9, R3.1, R4.2, R8.6 | `test_split.py::test_every_split_point_equals_the_single_process_model` | X1a at splits 1, 7, 15, 22, 29 (#17); capacity-aware split pending (#64) |
| Spec R4 | Transfer activations over RPC | R1.4, R3.4 | `test_chain.py::test_the_bytes_are_raw_little_endian_fp32_with_dtype_and_shape`, `test_protocol.py::test_services_match_the_agreed_rpcs` | X3 pending (#55) |
| Spec R5 | Per-request ordering | R3.7, R3.16, R4.2, R4.5 | `test_ordering.py::test_random_interleavings_through_both_stages_leave_no_violation_uncounted`, `test_concurrency.py::test_requests_interleave_and_each_stays_strictly_sequential` | X2 ordering count pending (#58) |
| Spec R6 | KV-cache state | R2.7, R3.5, R3.6, R3.8, R3.9, R4.6 | `test_kvstore.py`, `test_kv_replay.py::test_truncate_at_a_message_boundary_then_append_equals_a_fresh_run`, `test_concurrency.py::test_eight_requests_together_each_produce_what_they_produce_alone`, `test_heartbeat.py::test_frees_travel_on_the_heartbeat_and_empty_both_caches` | X5 pending (#60) |
| Spec R7 | Bounded request queues | R5.1, R5.2, R5.3, R7.14 | `test_backpressure.py::test_under_overload_the_excess_is_refused_at_once_and_every_bound_holds`, `test_kv_admission.py` | X7 pending (#59) |
| Spec R8 | Timeouts | R3.10, R5.4, R5.5, R5.6, R6.3, R6.9 | `test_deadlines.py::test_with_a_suspended_worker_every_request_ends_by_its_deadline`, `test_detection.py::test_a_suspended_stage_is_detected_by_its_heartbeat_timeout`, `test_fail_cleanly.py::test_a_worker_slower_than_the_step_timeout_fails_requests_at_their_deadline_within_bounds` | fault suite on both laptops (#46) |
| Spec R9 | Optional micro-batching | R5.7, R5.8, R5.9, R7.15 | none | optional clause, not built, #40 open |
| Spec R10 | Restart the affected worker | R3.3, R3.10, R3.13, R4.4, R6.2, R6.3, R6.4, R7.12 | `test_supervisor.py::test_a_worker_killed_by_the_fault_injector_is_respawned`, `test_detection.py::test_a_dead_epoch_waits_for_a_new_registration_and_a_route_update`, `test_control.py::test_a_stage_that_registers_again_gets_a_higher_epoch` | fault suite (#46); X5 pending (#60) |
| Spec R11 | Replay the prompt to rebuild state | R6.5, R6.6, R6.7, R6.18, R7.12 | `test_replay.py::test_the_replay_resends_the_original_messages_with_the_replay_flag`, `test_replay.py::test_a_killed_stage_process_is_replaced_and_the_request_finishes_unchanged`, `test_replay_cross.py` | fault suite (#46); X5 pending (#60) |
| Spec R12 | Discard stale outputs by request and step ids | R3.11, R3.12, R4.7, R4.8, R6.4, R6.8, R7.13 | `test_fencing.py`, `test_results.py::test_each_kind_of_mismatch_is_dropped_and_counted_by_its_reason`, `test_cross_result_filter.py`, `test_stale_after_suspend.py::test_a_resumed_old_stage_has_every_late_output_rejected` | fault suite (#46); X6 pending (#60) |
| Spec R13 | Compare with single-process inference | R1.14, R2.3, R2.4, R2.9, R7.8 | `test_model_weights.py::test_twenty_prompts`, `test_split.py::test_generation_through_the_split_equals_the_single_process_run`, `test_gateway.py::test_generate_gives_the_single_process_result_in_order` | X1a on both hosts (#17); X1b pending (#55) |
| Spec R14 | Report token latency | R7.4, R7.11 | `test_report.py::test_ttft_tpot_and_end_to_end_are_measured_from_arrival_at_the_gateway` | X2 pending (#58); X4 pending (#49) |
| Spec R15 | Report throughput | R7.5, R7.9 | `test_report.py::test_only_completed_requests_give_latencies_and_throughput` | X2 pending (#58) |
| Spec R16 | Report transferred bytes | R7.2, R7.10 | `test_metrics.py::test_sizes_are_the_serialized_sizes`, `test_x3.py::test_activation_bytes_are_hidden_size_x_4_per_position_within_the_overhead` | X3 pending (#55) |

Spec R9 has no test or run. It is an optional clause, not built (#40 open); if it is cut, R7.19 also cuts the X2 upper bound.

## 11. Evidence per claim and experiment

Author: joint. Each result comes from the script and command named beside it.

| Claim | Requirement | Test or script | Command | Result |
|---|---|---|---|---|
| Own forward pass equals the reference library | R2.3 | `tests/test_model_weights.py` | `uv run pytest tests/test_model_weights.py` | Maximum logit difference 0 and equal greedy tokens on 20 prompts, on macOS and Windows (#13) |
| X1a: the split equals the single process on each host | R2.9, R7.8 | `pllm.x1a` | `uv run python -m pllm.x1a smollm2-135m --threads <n> --length 128` | 250 of 250 exact token match, maximum logit difference 0.0, on each host: Host A commit cb385cd, 6 threads; Host B commit 86b3436, 3 threads (#17) |
| One request completes across both laptops | R8.1 | `pllm.launch`, `pllm.client` | `uv run python -m pllm.client "The capital of France is" --max-tokens 16 --timeout 30` | Ended at `max_tokens` after 16 tokens on the college Wi-Fi, 2026-11-26; 16 of 16 tokens agree with the Host A reference by decoded text (#54) |
| X1b: agreement across the laptops | R7.8 | `pllm.x1b` | `uv run python -m pllm.x1b smollm2-135m --metrics runs/<gateway>.metrics.jsonl --reference runs/<host A>-x1a.json runs/<host B>-x1a.json --rtt-ms <ms>` | pending (#55) |
| X2: throughput, TTFT, TPOT against concurrency | R7.9 | `pllm.x2`, `pllm.report` | `uv run python -m pllm.x2 pipeline smollm2-135m --metrics runs/<gateway>.metrics.jsonl` and `uv run python -m pllm.x2 single smollm2-135m` | pending (#58) |
| X3: bytes per hop | R7.10 | `pllm.x3` | `uv run python -m pllm.x3 smollm2-135m --length <16, 128 or 512> --metrics runs/<gateway>.metrics.jsonl` | pending (#55) |
| X4: latency against injected delay | R7.7, R7.11 | `pllm.x4` | `uv run python -m pllm.x4 smollm2-135m --metrics runs/<gateway>.metrics.jsonl` (gateway started with `--hop-delay-control`) | pending (#49) |
| X5: crash recovery of each stage | R7.12 | `pllm.x5` | `uv run python -m pllm.x5 smollm2-135m --metrics runs/<gateway>.metrics.jsonl --rtt-ms <ms>` | pending (#60) |
| X6: stale outputs after a suspend | R7.13, R6.8 | `pllm.x6`, `pllm.standby` | `uv run python -m pllm.x6 smollm2-135m --stages 1 --metrics runs/<gateway>.metrics.jsonl --rtt-ms <ms>` | pending (#60) |
| X7: overload with bounds on and off | R7.14 | `pllm.x7` | `uv run python -m pllm.x7 run smollm2-135m --metrics runs/<gateway>.metrics.jsonl --rate <requests per second>` | pending (#59) |
| X8: micro-batching | R7.15 | none | none | optional, not built (#40) |
| X9: capacity-aware split | R7.16, R8.6, R8.7 | `pllm.x9`, `pllm.capacity` | `uv run python -m pllm.x9 run smollm2-135m --metrics runs/<gateway>.metrics.jsonl --budget-mib 716.8 --rtt-ms <ms>` (after `probe` and `choose`, README "X9") | pending (#64) |
| Fault suite: every failure-handling claim has a test that injects the fault | R6.13 | tests marked `fault` | `uv run pytest -m fault -rs` | Host B: 129 passed, 1 skipped (needs 1.5 GiB of free memory), 0 failed, commit 86b3436. Host A: 102 passed, 0 skipped, commit not stated; PR #135 says this run came before the cross-tests were marked, and on #46 Member B noted it may be from an earlier commit. The two laptops ran different test sets; a Host A run at 86b3436 would give one set (#46) |
| Recovery time in four parts (detection, respawn, weight load, replay prefill) | R6.12 | `recovery` lines in the metrics file, `pllm.report` | from the X5 runs | pending (#60) |

Fault suite claims and their tests:

| Claim | Fault injected | Tests |
|---|---|---|
| R6.1 kill, suspend, resume, per-step sleep | kill and suspend of real process trees; a step delay | `test_faults.py`, `test_supervisor.py` |
| R6.2 supervisor respawns, respawn time logged | kill; a worker that exits or crashes | `test_supervisor.py` |
| R6.3 detection by stream break, heartbeat timeout, step deadline | kill, suspend, a slow step | `test_detection.py`, `test_detection_cross.py` |
| R6.4 dead epoch, new registration, route update | a stopped or resumed stage | `test_detection.py`, `test_detection_cross.py` |
| R6.5 idle requests holding KV are bumped | a stopped stage while a request waits for a credit | `test_replay.py`, `test_replay_cross.py` |
| R6.6 chunked replay through both stages | a stopped stage; a GAP Nack | `test_replay.py`, `test_replay_cross.py` |
| R6.7 replay verified, output equals a fault-free run | kill of a real stage process; a forced mismatch | `test_replay.py`, `test_replay_cross.py` |
| R6.8 late outputs of a resumed stage rejected | suspend past the timeout, resume after the replacement | `test_stale_after_suspend.py` |
| R6.9 attempts, MAX_DEATHS, growing pause, deadline | repeated deaths; a refused replay; no replacement | `test_fail_cleanly.py`, `test_replay.py`, `test_replay_cross.py` |
| R6.10 a slow worker fails cleanly at the deadline | a step delay; a suspended worker | `test_fail_cleanly.py`, `test_deadlines.py` |
| R6.11 a delayed or lost hop | one message lost, or held past the step timeout or past the deadline, on each hop | `test_hop_faults.py::test_a_lost_hop_message_fails_its_request_at_the_deadline_and_holds_up_no_other`, `::test_a_hop_message_late_past_the_step_timeout_still_completes_its_request`, `::test_a_hop_message_late_past_the_deadline_fails_its_request_and_its_result_is_dropped` |
| R6.12 recovery time in four parts | kill of a real stage process; suspend and resume | `test_recovery.py`, `test_stale_after_suspend.py` |
| R6.14 both stages at once | none yet | pending (#52) |

## 12. How to run

The README is the reference; this section only points to it.

- Install and test: README "Development" (`uv sync`, `uv run pytest`).
- Weights: README "Model weights" (`uv run python -m pllm.models smollm2-135m`).
- Two-laptop session: README "Session checklist", then "Host checks" for the network probe, then "Starting the system".
- Each experiment: the README section of that name (X1a, X1b, X2, X3, X4, X5 and X6, X7, X9).
- Tables and plots: README "Report" (`uv run python -m pllm.report`).

## 13. Conclusion

Author: joint. To be written once the results in section 9 are in.
