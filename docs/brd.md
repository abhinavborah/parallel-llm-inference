# Business Requirements Document: Pipeline-Parallel LLM Inference with Fault Recovery

**Project:** parallel LLM inference (course topic 38)
**Version:** 1.2 (draft for review by both members)
**Date:** 2026-10-26
**Authors:** project team (two members)
**Baseline:** the scope proposal submitted to the course TA. Where this document and the scope proposal differ, the scope proposal governs until both members agree a change.

## 1. Purpose

This document states what the project must deliver, how success is judged, and how the work is split into epics and user stories on a GitHub Project board. It turns the scope proposal and the course specification into rows that can be tracked, and it records the process rules (gates, test lanes, reproducibility, Definition of Done) that protect the validity of the results. Stories are derived from the requirement rows in section 8.

## 2. Background

Large language models are often too big or too slow for one device. Model parallelism splits layers across machines and passes activations between them. The course specification asks for pipeline inference of a small supplied language model across two worker processes, with ordered per-request state, bounded queues, timeouts, optional micro-batching, restart and prompt replay on failure, stale-output discarding, and a comparison with single-process inference.

The project builds this on two laptops the team already owns. Stage 0 (embedding and the first layers) runs on a Windows laptop with about 2.5 GB usable for the worker. Stage 1 (remaining layers, final norm, LM head, greedy argmax) and the gateway run on an ARM64 macOS laptop with 24 GB. Activations travel worker to worker over gRPC. The model would fit on one machine; it is split to demonstrate the distributed mechanisms, and the smaller laptop forces a memory-aware split.

The central claims to demonstrate are: tokens equal single-process output (bit-exact on one host, measured agreement across hosts); every token is delivered to the client exactly once even though computation is at-least-once; stale outputs from a dead or paused worker incarnation are fenced out by request, attempt, step and epoch; and backpressure keeps queues and memory bounded.

### Terms

| Term | Meaning |
|---|---|
| Gateway | Process that accepts client requests, applies admission control, drives the decode loop, owns the authoritative token history, assigns epochs, filters results and records metrics. |
| Stage 0, Stage 1 | The two worker processes. Stage 0: embedding and layers 0 to k-1. Stage 1: layers k to n-1, final norm, LM head. |
| Supervisor | One per laptop. Spawns and respawns its worker and executes fault-control commands. |
| Host A, Host B | Host A is the macOS laptop (gateway, Stage 1). Host B is the Windows laptop (Stage 0). |
| rid, attempt, step, epoch | Request id; replay generation of a request; decode step index; incarnation number of a stage assigned by the gateway at registration. Together they fence stale outputs. |
| KV cache | Per-request key and value tensors, indexed by absolute position, held on each stage. |
| Replay | After a stage dies, the gateway resends the prompt and the already-emitted tokens so the new incarnation rebuilds KV state. |
| TTFT, TPOT | Time to first token; time per output token. Both measured at the gateway. |
| Requirement ids | Written R<epic>.<n>, for example R6.3. Course specification clauses are written "Spec R1" to "Spec R16", always with the prefix "Spec". |
| Proposal goals | PG1 to PG6, the six goals in the scope proposal: pipeline model parallelism; correct ordering and state; backpressure and flow control; failure recovery; correctness validation; measurement. |
| Experiments | X1a, X1b, X2 to X10 are experiment ids. X<n> is an experiment; E<n> is an epic. Epic ids (E1 to E10 in section 8) are written "Epic E<n>" in prose. |
| Decisions | D1 to D12 are the team decisions numbered in the scope work; N1 to N8 are questions first raised in this document; TA1 to TA4 are questions for the course TA. |

## 3. Scope

**In scope**
- A two-stage pipeline: gateway, two workers, a supervisor on each laptop, with gRPC protocol definitions, bounded queues, timeouts, and per-request KV state.
- A minimal Llama-family forward pass written in PyTorch (CPU, fp32) with explicit position-indexed KV tensors, loading safetensors weights. Hugging Face transformers is a pinned test oracle only.
- Models: SmolLM2-135M-Instruct for development, tests and, for now, the presented runs (#61); Qwen3-0.6B stays pinned and tested for headline runs if time and the Host B budget allow (needs QK-norm).
- Epoch-fenced crash recovery with prompt replay, fault injection (kill, suspend, delay) and a stale-output filter.
- Optional micro-batching (decode only).
- Capacity-aware layer split chosen from each host's free memory and speed.
- A test suite (single-process equivalence, KV truncation and replay, stale-output rejection, queue overflow), experiments X1a to X9, a final report with plots and limitations, and a live demo on both laptops.

**Out of scope**
- Gateway failure, Byzantine workers, strong security. These are stated in the report as limits of the fault model.
- GPU execution for equality runs (a device flag exists; Host B has no usable GPU).
- Replacing gRPC with torch.distributed (its elastic launchers restart all workers on one failure, which conflicts with restarting only the affected worker).
- Production deployment, autoscaling, quantisation, sampling other than greedy argmax.

**Stretch (only after every Must exit criterion passes)**

| Item | Epic | Note |
|---|---|---|
| Web dashboard that sends requests and shows live state | Epic E9 | The scope proposal lists it as a stretch goal. A weekly schedule exists for it, so its status needs both members' agreement (D11). |
| Service discovery (mDNS) and 6-digit code pairing, manual IP entry kept as default | Epic E9 | Same status as the dashboard. |
| Partial replay (experiment X10): Stage 0 keeps outbound activations so a Stage 1 failure replays only Stage 1 | Epic E6 | Decision D6 defaults to full replay. |
| Gateway durability: durable request log so the gateway can restart | Epic E6 | Decision D5 defaults to out of the fault model. |
| Qwen3-1.7B with the stage role swap, if Host B memory allows | Epic E8 | Pending measured memory. |

## 4. Stakeholders and roles

| Role | Interest |
|---|---|
| Member A (macOS host, Host A) | Builds and owns the components listed for A in section 8. Runs the gateway and Stage 1 in every LAN run. Reviews Member B's pull requests. |
| Member B (Windows host, Host B) | Builds and owns the components listed for B in section 8. Runs Stage 0 in every LAN run, owns the Windows setup and firewall. Reviews Member A's pull requests. |
| Course TA and evaluators | Review the proposal, the submission and the demo. Judge whether each clause of the course specification is met and evidenced, and how the work was shared. |

Division of work is balanced by estimated effort (A about 14 days, B about 15 days, plus about 4 days of unassigned dashboard and discovery work). Owners in section 8 are written "A (unconfirmed)" or "B (unconfirmed)" until both members confirm the split. Both members must be able to explain the whole system, so each writes the tests for one of the other's components (R3.14, R4.9).

## 5. Objectives and success measures

| ID | Objective | Measure |
|---|---|---|
| O1 | A working two-stage pipeline across two laptops. | Two OS processes, each holding only its own layers; one request generates end to end over the LAN; same code path on macOS and Windows. |
| O2 | Output equals single-process inference. | X1a: 100 percent exact token match on each host (batch 1, fp32, fixed thread count) for 50 prompts at every tested split. X1b: token agreement rate, first divergence position and maximum logit difference reported for the two-laptop run. |
| O3 | Per-request ordering and KV consistency. | Zero ordering violations in all X2 runs; equality holds when 8 requests interleave. |
| O4 | Backpressure holds. | Queue depth never exceeds its bound; p99 latency bounded under overload; Host B memory stays within its stated budget (X7). |
| O5 | Failure recovery with fencing. | After a kill or suspend of each stage mid-generation with several requests in flight, every request completes with output equal to a fault-free run (batch 1); zero duplicate tokens at the client; every stale output rejected; each recovery increments the attempt counter; an unrecoverable request fails at its deadline (X5, X6). |
| O6 | Communication cost is understood. | Measured bytes per token per hop equal hidden size x 4 B x tokens within framing overhead (X3); effect of 0, 1, 5, 20 and 50 ms injected delay reported (X4). |
| O7 | Performance is reported honestly. | TTFT and TPOT at p50 and p95, throughput in tokens per second against concurrency 1 to 16, with at least 5 repeats per LAN run. Single-request latency is expected to be higher than single-process; throughput gain with concurrency is expected up to about 2x over the pipeline at concurrency 1, and is not promised. |
| O8 | Memory-aware split. | Balanced split of the presented model (SmolLM2-135M for now, #61) compared with an even split on throughput and Host B peak memory; Host B does not page to disk (X9). |
| O9 | Deliver on time with evidence. | Report, plots and rehearsed demo by the final week; every row Spec R1 to Spec R16 has cited evidence; every reported number comes from a script. |
| O10 | (Stretch) Make behaviour visible to evaluators. | Dashboard shows epochs, replay and rejections live; pairing and discovery demonstrated; manual IP entry always works. |

## 6. Key dates

The plan assumed a start on 2026-10-12 and 45 days, ending about 2026-11-25. The course has not released the dates; the probable submission deadline is now 2026-11-15 and the demo is probably around 2026-11-20, both reported by Member B and not confirmed by the course (N8, R10.1, #70). The table below keeps its original dates: W6 and the buffer fall after 2026-11-15 and are to be pulled forward, and no story is cut. Shift this table once the real dates are known. A week is not done until its exit criteria pass.

| Week | Dates | Deliverables | Exit criteria | Status on 2026-10-26 |
|---|---|---|---|---|
| W1 | Oct 12 to 18 | Repository, uv lockfile on both OSes; own Llama forward for SmolLM2-135M; single-process reference; in-process two-stage split; protocol schema frozen; fault injector skeleton | Own forward matches the pinned reference library (max logit difference under 1e-4, same tokens on 20 prompts); in-process split equals single-process exactly; prefill chunking test decides replay shape | Not met. Code has not started. |
| W2 | Oct 19 to 25 | gRPC stages in chain topology; gateway with one request at a time; byte counters; JSON-lines metrics; LAN bring-up with Stage 0 on Windows and Stage 1 plus gateway on macOS | X1a passes on each laptop; one request generates end to end across both laptops; X1b and X3 numbers produced | Not met. |
| W3 | Oct 26 to Nov 1 | Many requests in flight; per-request KV store; ordering asserts; bounded admission queue; credit limits; deadlines on every hop; dashboard v1 (stretch) | X2 and X7 run end to end; zero ordering violations | Starts today. |
| W4 | Nov 2 to 8 | Heartbeats, supervisor respawn, epochs, attempts, replay with truncate, stale-output filter | X5 and X6 pass: output equals reference after crash; zero duplicate tokens after pause | Planned. |
| W5 | Nov 9 to 15 | Micro-batching; delay injection; capacity-aware split sweep; switch presented runs to Qwen3-0.6B (deferred by #61); discovery and pairing (stretch) | X4, X8, X9 produce data; Host B never pages | Planned. |
| W6 | Nov 16 to 22 | Full experiment runs with repeats; plots; report draft; demo rehearsed twice | Every row Spec R1 to Spec R16 has evidence | Planned. |
| Buffer | Nov 23 to 25 | Fixes; stretch only if all exit criteria pass | Submission | Planned. |

The week column in section 8 shows the roadmap week in which a requirement was planned. Rows marked W1 or W2 are overdue today.

**Landed.** The repository, uv lockfile, pytest and ruff setup, and CI on Linux, macOS and Windows landed on 2026-10-26 (R1.2, R1.3, and part of R1.1). No feature code exists yet.

**Slip.** The project is two weeks behind. W1 and W2 deliverables (the oracle, the frozen protocol, the gRPC chain and the first LAN run) do not exist, and W3 starts today. Four weeks of planned time remain (W3 to W6) plus a 3-day buffer, for six weeks of planned work. The risk order in the plan is kept: the oracle and fault injector come before features, and recovery (mandatory) comes before micro-batching (optional).

**Recovery note (proposed, not yet agreed by both members)**

| Dates | Target | Notes |
|---|---|---|
| Oct 26 to 28 | W1 exit: oracle, in-process split, protocol frozen, lockfile on both OSes, fault injector skeleton | A and B work in parallel: B on forward, weights and oracle; A on protocol and gateway skeleton. |
| Oct 29 to Nov 1 | W2 exit: gRPC chain, LAN bring-up, X1a, one end-to-end request, byte counters, X3 | Run the network check (R1.8) and the Host B memory measurement (R1.7) on the first day. |
| Nov 2 to 8 | W3 and W4 core: concurrency, bounded queues, ordering asserts, heartbeats, epochs, replay, fencing; X7, X5, X6 | Fault tests written with each feature, not afterwards. |
| Nov 9 to 15 | W5 reduced: delay injection, capacity-aware split, Qwen3-0.6B switch (deferred by #61); X4, X9; micro-batching if time remains | |
| Nov 16 to 22 | W6 as planned: repeats, plots, report, demo rehearsed twice | |
| Nov 23 to 25 | Buffer | Submission. |

If progress falls behind this table, cut in this order: all of Epic E9 and the other stretch items; micro-batching and X8; the X9 split sweep reduced to the even and balanced splits only; the X4 delay sweep reduced to 0, 5 and 50 ms; Qwen3-0.6B headline (stay on SmolLM2-135M and say so). Never cut: the equality oracle, fencing, replay, the backpressure tests, and the fault-injection test for each failure-handling claim. Reducing repeats below 5 per LAN run needs both members' agreement and a note in the report. Slips are flagged to the other member the day they are seen, and this table is re-checked at the end of each week.

## 7. Process and quality requirements

### 7.1 Gates (engineering invariants)

These are blocking. A story that would break one is not Done.

| Gate | Rule |
|---|---|
| G1 | The correctness oracle comes first: the single-process reference and equality tests exist before features build on them. |
| G2 | Every failure-handling claim (restart, replay, fencing, backpressure) has an automated test that injects the fault. |
| G3 | Bit-exact equality is claimed only on one host, fp32, batch 1. Cross-host results are reported as agreement (rate, first divergence position, maximum logit difference). |
| G4 | Latencies are measured at the gateway with one clock. Timestamps from two machines are never subtracted. Reported numbers come from scripts, never from the dashboard. |
| G5 | Host B has about 2.5 GB for the worker. Any change that raises its memory use states the new budget. |
| G6 | Python with uv; one lockfile for both operating systems; the same PyTorch version on both hosts. |
| G7 | Everything runs on both macOS and Windows. Scripts are written in Python, not shell-specific tools. |
| G8 | Before changing an algorithm, model, library or metric, current practice is researched with dated sources and reported to both members before the decision. |
| G9 | Non-trivial work is planned and agreed before it starts. Deadlines are hard; a slip is flagged early. |

### 7.2 Test lanes

| Lane | What it covers | Where it runs |
|---|---|---|
| Unit | Protocol encode and decode; stage acceptance rules (epoch, sender, attempt, start_pos); KV truncate-then-append; gateway result filter; admission queue and credit limits; request table; replay message sequencing; split computation. Uses fakes for the transport and tiny or synthetic weights. | CI on every pull request, on macOS, Windows and Linux runners. |
| Single-host integration | All processes on one laptop. X1a exact equality (reference against pipeline); kill, suspend and delay fault tests; interleaving of 8 requests; overload test. SmolLM2-135M on both laptops; larger models only on Host A. | Each laptop by hand or by a slow-test marker; not in CI if weights are required. |
| Two-laptop LAN | X1b and X2 to X9 with Stage 0 on Host B and Stage 1 plus gateway on Host A. At least 5 repeats per run. Each run records network, round-trip time, available memory and power state. | By script on the two laptops. Never in CI. |

- Cross-testing: each member writes tests for a component the other built (R3.14, R4.9).
- Property-style tests are used where an implementation could fit visible examples only (for example, random interleavings of start_pos messages).

### 7.3 Reproducibility

- uv lockfile shared by both OSes; same PyTorch version on both hosts; Python 3.12.
- Models are pinned by repository revision and checked by checksum at registration (weights_sha256).
- Every run writes a record: commit hash, model and revision, split point, thread count, host, OS, network used, round-trip time, available memory at start, power state, date.
- All reported numbers and plots come from scripts that read JSON-lines run logs. The dashboard is a demo aid only.
- Large artifacts (model weights, run outputs, plot data) are not committed. Committed summary tables and the scripts that regenerate them are.

### 7.4 Workflow and Definition of Done

- Epics map to the ten epics in section 8. Stories are derived from requirement rows. Board statuses: Backlog, Ready, In progress, In review, Done. One story per branch and pull request; the pull request body has `Closes #N`. Rows labelled "Decision:" become issues of type decision, and each blocks the stories named in its row.
- Short-lived branch per story, named `e<epic>-s<story>-<slug>` (lowercase, hyphens); non-story work uses `fix/`, `chore/` or `ci/`. Nothing is committed to `main` directly.
- Commit messages: conventional prefix (`feat:`, `fix:`, `test:`, `ci:`, `chore:`, `docs:`), lowercase, neutral. One logical change per commit. Each member commits under their own identity.
- Pull request reviewed and approved by the other member before merge; merge with a merge commit, then delete the branch. Until the second member has access to the repository, the author may merge their own pull request, and the pull request body says so.

A story is Done when all of these hold:
1. Its acceptance criteria pass.
2. CI is green on macOS, Windows and Linux.
3. The other member has reviewed and approved it (or the author merged under the interim rule above).
4. Deterministic code has unit tests.
5. Each failure-handling claim in the story has a fault-injection test (gate G2).
6. An experiment story states its metric, sweep and pass criterion; any reported number comes from a script whose command is recorded, and its run record exists.
7. If the change raises Host B memory use, the new budget is stated in the pull request (gate G5).
8. Wire or protocol changes are agreed by both members and the protocol schema version is updated.

## 8. Requirements by epic

Priority: Must (P0 on the board), Should (P1), Could (P2). Week: the roadmap week in which the item was planned (section 6). Owner: A is Member A (macOS host), B is Member B (Windows host); every owner is unconfirmed until both members agree the split. Source: Spec R<n> is a course specification clause; "Proposal" with a goal id is the scope proposal; E<n> as experiment id, D<n>, N<n> and TA<n> as in the Terms table.

### Epic E1. Foundations and environment

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R1.1 | Repository layout and a README that states how to install, run the tests, and start each process on macOS and Windows. Status: partly done (merged 2026-10-26: README Development section with install, test and lint commands); process start instructions remain. | Must | W1 | A and B (unconfirmed) | Proposal PG1; G7 |
| R1.2 | One uv lockfile that resolves on macOS ARM64 and Windows x86-64 with Python 3.12 and the same CPU PyTorch version; `uv sync` succeeds on both laptops. Status: Done (merged 2026-10-26: uv project, lockfile, pytest and ruff). | Must | W1 | B (unconfirmed) | G6; Proposal tools section |
| R1.3 | Continuous integration that runs lint and the unit lane on every pull request, green on macOS, Windows and Linux runners. Status: Done (merged 2026-10-26). | Must | W1 | A (unconfirmed) | Section 7.2 |
| R1.4 | Protocol schema frozen: services Control (Register, Heartbeat), Stage (Steps), Results (Push), Frontend (Generate) and the messages in the agreed schema; generated code builds on both OSes; later changes need both members. | Must | W1 | A and B (unconfirmed) | Spec R4; Proposal PG1; D4 |
| R1.5 | Decision D4: transport is gRPC with raw tensor bytes (default) or ZeroMQ. Blocks R1.4. Status: decided 2026-11-01 (#4): gRPC with raw tensor bytes. | Must | W1 | A and B (unconfirmed) | D4 |
| R1.6 | A configuration file holding both host addresses and ports, read by every process; no address is hard-coded. Stays supported when discovery exists. | Must | W1 | A (unconfirmed) | Proposal network section |
| R1.7 | Measure Host B available memory with other programs closed (Task Manager, method and date recorded) and state the worker budget. About 2.5 GB is assumed until measured. Feeds R5.3 and R8.6. Status: measured 2026-11-03 (#7) with other programs closed and the charger in. `uv run python -m pllm.hostmem` read psutil available memory once a second for 60 s: minimum 1.28 GiB, median 1.55 GiB of 5.85 GiB total, 0.62 GiB of swap in use. Task Manager at the same time showed 4.4 of 5.8 GB In use (its GB are GiB); total minus In use is about 1.4 GiB, close to the script but not the same quantity: it also contains the modified pages, which Available leaves out, and the two readings were not taken at the same instant. The next run records Task Manager's Available figure directly. A run on 2026-11-07 with other programs open read a minimum of 0.73 GiB and a median of 0.79 GiB, so 1.28 GiB is the clean best case, not a typical reading; a clean second run (for example after a restart) is still to be done. Proposed worker budget 0.7 GiB (minimum less 0.5 GiB headroom), unconfirmed by Member A. The budget is the peak RSS of the whole worker process: the Python and PyTorch runtime (about 196 MiB on Host B, the RSS of a fresh process after importing torch and the model code, measured on 2026-11-07 and again on 2026-11-09 at 194 to 195 MiB over three runs with `uv run python -c "import psutil, torch, pllm.model; print(psutil.Process().memory_info().rss / 2**20)"`; a check, not a recorded run), the Stage 0 layers and the KV cache. The KV budget that R5.3 admission uses is derived from it by subtracting the runtime and the weights. The 0.5 GiB headroom is a proposed margin, not a measurement: it covers programs that start during a run and memory the 60 s sample did not see, and it is unconfirmed like the budget. On the 2026-11-07 reading (minimum 0.73 GiB) the same headroom would leave 0.23 GiB, so the budget may not hold when other programs are open; R5.3 and R8.7 depend on the second clean run. This is well below the 2.5 GB assumed in G5, K2, R5.3 and R8.6, which stay as written until both members agree how to respond. | Must | W1 | B (unconfirmed) | D8; Proposal special requirements |
| R1.8 | Network check between the laptops: TCP in both directions on the worker port, round-trip time, Windows network profile and firewall state (for now Public with the firewall off, agreed 2026-11-09; with the firewall on, an inbound rule for the worker port on the profile in use), multicast test; result recorded. Fallbacks in order: direct cable, phone hotspot, Tailscale. | Must | W2 | A and B (unconfirmed) | Proposal special requirements |
| R1.9 | Script that downloads the pinned model files (SmolLM2-135M-Instruct, Qwen3-0.6B) with checksum verification, into a folder that is not tracked by git. | Must | W1 | B (unconfirmed) | Proposal PG5; G6 |
| R1.10 | Run-record writer used by every script (fields in section 7.3); `.gitignore` excludes weights and run outputs. | Must | W1 | A (unconfirmed) | Section 7.3 |
| R1.11 | Decision TA1: is the model ours to choose or supplied? Blocks the final model choice in R1.12. Status: decided 2026-11-01 (#8): the model is ours to choose. | Must | W1 | A and B (unconfirmed) | TA1; Spec R1 |
| R1.12 | Decision D1: confirm SmolLM2-135M for development and Qwen3-0.6B as the headline model, and use one wording for it across all project documents. Fallbacks: SmolLM2-360M, GPT-2, Qwen3-1.7B with role swap. Status: decided 2026-11-03 (#9): SmolLM2-135M-Instruct for development and tests, Qwen3-0.6B for headline runs; fallback SmolLM2-360M if a Qwen3-0.6B Stage 0 does not fit Host B after #7, Qwen3-1.7B stays stretch, GPT-2 dropped. Update 2026-11-07 (#61): presented runs use SmolLM2-135M until the pipeline, recovery and the main experiments work end to end; Qwen3-0.6B stays pinned and tested and the presented runs switch back to it if time and the Host B budget allow. | Must | W1 | A and B (unconfirmed) | D1; Proposal goals |
| R1.13 | Decision TA2: is a gateway as a third process acceptable alongside the two workers? If not, the gateway role moves into a worker. Status: decided 2026-11-01 (#10): the gateway is a separate process; the specification limits only the workers. | Should | W1 | A and B (unconfirmed) | TA2; Spec R1 |
| R1.14 | Decision TA3: is bit-exact equality per host, with cross-host equality reported as agreement, acceptable? Status: decided 2026-11-01 (#11): bit-exact per host at batch 1, agreement across hosts. | Should | W1 | A and B (unconfirmed) | TA3; Spec R13; G3 |

### Epic E2. Reference model and equality oracle

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R2.1 | Minimal Llama-family forward pass (RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings) in CPU fp32 PyTorch, loading safetensors weights directly, with explicit position-indexed KV tensors. | Must | W1 | B (unconfirmed) | Spec R1; Proposal PG1 |
| R2.2 | QK-norm support so Qwen3-0.6B loads and runs. | Must | W3 | B (unconfirmed) | D1; Proposal tools section |
| R2.3 | Test: the own forward matches the pinned reference library on logits (max absolute difference under 1e-4) and on greedy tokens for 20 prompts. The reference library is used in tests only. | Must | W1 | B (unconfirmed) | Spec R13; Proposal PG5 |
| R2.4 | Single-process reference runner using the same forward code: tokenise, prefill, greedy decode, output tokens and top logit per step; thread count set by a parameter. | Must | W1 | B (unconfirmed) | Spec R13; G1 |
| R2.5 | In-process two-stage split: layers 0 to k-1 with the embedding, layers k to n-1 with final norm and LM head; k configurable. Output equals single-process exactly for every k tested. | Must | W1 | B (unconfirmed) | Spec R3; X1a |
| R2.6 | Fixed evaluation prompt set: 50 prompts at lengths 16, 128 and 512 tokens, stored as a checksummed file and used by every experiment. | Must | W1 | B (unconfirmed) | Proposal evaluation setup |
| R2.7 | Unit test: truncating a KV cache to position p and appending equals a fresh run to the same length. | Must | W1 | B (unconfirmed) | Spec R6; Proposal PG2 |
| R2.8 | Test: is the KV state after prefill of P+m tokens bitwise equal to prefill of P followed by m single-token decodes? The result decides whether single-message replay is allowed (R6.6). | Must | W1 | B (unconfirmed) | Roadmap exit W1; Proposal PG4 |
| R2.9 | X1a harness: 50 prompts x splits {1, n/4, n/2, 3n/4, n-1}, run on each laptop alone, reporting exact token match and maximum logit difference as JSON lines. Pass: 100 percent match on each host. | Must | W2 | B (unconfirmed) | X1a; Spec R3; Spec R13 |
| R2.10 | A device flag with CPU as the default and the only device used in equality runs. | Should | W1 | B (unconfirmed) | Spec R2; D3 |
| R2.11 | Decision N1: if the own Llama-family forward does not match the reference by the end of W1, ship on GPT-2 small. Every distributed-systems requirement is model independent. Blocks the W2 schedule. Status: not needed 2026-11-01 (#13): the own forward matches the reference (#12). | Should | W1 | A and B (unconfirmed) | D1; model fallback |
| R2.12 | Store single-process reference outputs (tokens and logit summary) per host with a checksum, so X1b compares against both hosts' references. | Should | W2 | B (unconfirmed) | X1b |

### Epic E3. Stage workers and protocol

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R3.1 | Stage worker process hosting the Stage service. It loads only its layers (range from configuration) and logs weight load time and RSS at start. Two OS processes, each holding only its layers. | Must | W2 | B (unconfirmed) | Spec R1; Spec R3 |
| R3.2 | Stage 0 reads the embedding through a memory-mapped file so only touched rows enter RAM; Stage 1 holds the final norm and the LM head. Test: RSS stays under the Host B budget while embedding tokens. | Must | W2 | B (unconfirmed) | D9; Proposal special requirements |
| R3.3 | Register: a stage sends stage index, layer range, model id, weights hash and address; the gateway rejects a mismatched model or hash and returns an epoch. | Must | W2 | B (unconfirmed) | Spec R10; Proposal PG4 |
| R3.4 | Chain data path gateway to Stage 0 to Stage 1 to gateway over gRPC streaming RPCs. Activations are raw little-endian fp32 bytes with dtype and shape, never JSON. | Must | W2 | B (unconfirmed) | Spec R4; Proposal PG1 |
| R3.5 | Per-request KV store on each stage, indexed by position, keyed by request id, with a defined memory cost per request. | Must | W2 | B (unconfirmed) | Spec R6; Proposal PG2 |
| R3.6 | start_pos rules: equal appends; lower truncates then appends; higher is rejected with a GAP Nack so the gateway triggers replay. One unit test per case. | Must | W2 | B (unconfirmed) | Spec R6; Proposal PG2 |
| R3.7 | Step ordering asserted on every hop: step is monotonic per (rid, attempt). A violation is counted and logged. | Must | W3 | B (unconfirmed) | Spec R5; Proposal PG2 |
| R3.8 | Test: 8 interleaved requests give the same tokens as each request run alone. | Must | W3 | B (unconfirmed) | Spec R6; X2 |
| R3.9 | Free message releases a request's KV on both stages; test that the cached request count returns to 0. | Must | W3 | B (unconfirmed) | Spec R6; Proposal PG3 |
| R3.10 | Heartbeat bidirectional stream: each stage sends its stage, epoch, inbox depth and cached request count at a fixed interval; the gateway sends route updates and free messages back. | Must | W4 | B (unconfirmed) | Spec R8; Spec R10 |
| R3.11 | Stage fencing: reject a wrong target epoch (WRONG_EPOCH), a sender epoch that differs from the live upstream epoch (BAD_SENDER), and an older attempt (STALE_ATTEMPT); adopt a newer attempt. One unit test per rule. | Must | W4 | B (unconfirmed) | Spec R12; Proposal PG4 |
| R3.12 | Every Nack travels on the heartbeat stream so the gateway learns of a rejection immediately rather than by timeout. | Must | W4 | B (unconfirmed) | Spec R12 |
| R3.13 | Route update: Stage 0 receives the address and epoch of a restarted Stage 1 and reconnects without restarting. | Must | W4 | B (unconfirmed) | Spec R10 |
| R3.14 | Cross-test: the KV truncate and fencing tests for the stage worker are written by Member A, not the implementer. | Must | W4 | A (unconfirmed) | Contribution rule; G2 |
| R3.15 | Worker code is the same on both OSes; OS-specific code is limited to the supervisor and fault injector. | Must | W2 | B (unconfirmed) | Spec R2; G7 |
| R3.16 | Decision N2: the specification wording for ordering says one bidirectional stream per hop, while the protocol uses client-streaming for steps and results and bidirectional only for heartbeat. Both members choose one wording. Status: decided 2026-11-01 (#19): client-streaming for steps and results, bidirectional for heartbeat, as in the scope proposal. | Should | W2 | A and B (unconfirmed) | Spec R5; Proposal tools section |
| R3.17 | Decision N3: gRPC's default 4 MiB receive limit rejects a 1,024-token Qwen3-0.6B prefill (4 MiB plus framing). Either raise the limit on both channels or chunk long prefills; chunking interacts with R6.6. Prompts of 16, 128 and 512 tokens fit. Status: decided 2026-11-03 (#20): no chunking; every server and channel sets the receive limit from config, sized from batch, prompt length and hidden size, and the gateway rejects prompts above it. | Should | W2 | A and B (unconfirmed) | Protocol size analysis |

### Epic E4. Gateway and decode loop

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R4.1 | Frontend Generate(prompt, max_tokens, deadline) returning a server stream of tokens in order. | Must | W2 | A (unconfirmed) | Proposal PG1 |
| R4.2 | Gateway tokenises the prompt, assigns a request id and attempt 1, sends the prompt as one message (start_pos 0), then drives greedy decode one token per step. | Must | W2 | A (unconfirmed) | Spec R3; Spec R5 |
| R4.3 | Request table: request id, attempt, tokens so far (authoritative history), expected step, deadline. | Must | W2 | A (unconfirmed) | Proposal PG2; PG4 |
| R4.4 | Control service Register assigns a monotone epoch per stage incarnation. | Must | W2 | A (unconfirmed) | Spec R10 |
| R4.5 | Many requests in flight: each request is strictly sequential, requests interleave freely, no cross-request ordering promise. | Must | W3 | A (unconfirmed) | Spec R5; Proposal PG2 |
| R4.6 | A request ends on end-of-sequence or max_tokens, after which the gateway frees its state on both stages. | Must | W2 | A (unconfirmed) | Spec R6 |
| R4.7 | Result filter: accept a result only when attempt equals the current attempt, step equals the expected step and epoch equals the live epoch of the last stage; otherwise drop it and count it by reason. | Must | W4 | A (unconfirmed) | Spec R12; Proposal PG4 |
| R4.8 | Exactly-once delivery: tokens are streamed to the client once, in order, deduplicated on (rid, step). Test with an injected duplicate and an injected late result. | Must | W4 | A (unconfirmed) | Spec R12; X6 |
| R4.9 | Cross-test: the result-filter and replay tests for the gateway are written by Member B, not the implementer. | Must | W4 | B (unconfirmed) | Contribution rule; G2 |

### Epic E5. Flow control and micro-batching

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R5.1 | Fixed-size admission queue at the gateway; when full, a new request is rejected immediately with RESOURCE_EXHAUSTED. | Must | W3 | A (unconfirmed) | Spec R7; Proposal PG3 |
| R5.2 | Credit limit on in-flight steps per stage and a bounded stage inbox; queue depth never exceeds its bound under overload (test). | Must | W3 | A (unconfirmed) | Spec R7; X7 |
| R5.3 | Memory-aware admission: a request is admitted only if Host B's KV budget (from R1.7) can hold it, computed from concurrency x context x per-layer KV bytes. | Must | W3 | A (unconfirmed) | Spec R7; Proposal PG3 |
| R5.4 | Per-request deadline propagated on every hop. | Must | W3 | A (unconfirmed) | Spec R8; Proposal PG3 |
| R5.5 | Per-step timeout at the gateway. An expired step starts the failure path (Epic E6) or fails the request at its deadline. | Must | W3 | A (unconfirmed) | Spec R8 |
| R5.6 | No request hangs: every request ends completed, rejected, or failed with a deadline error within its deadline plus a stated tolerance. Test with a suspended worker. | Must | W3 | A (unconfirmed) | Spec R8; X6 |
| R5.7 | Micro-batching in the stage scheduler: collect up to B decode steps within a window of W ms, pad the KV with an attention mask. Decode only: a message with length above 1 (a prompt or a replay prefill) runs alone at batch 1; replay steps run unbatched and fault runs stay at batch 1 (R6.7). | Should | W5 | B (unconfirmed) | Spec R9; D7 |
| R5.8 | Batch size B in {1, 2, 4, 8} and window W in {0, 2, 5} ms are configurable; B = 1 with W = 0 behaves as the non-batched path. | Should | W5 | B (unconfirmed) | Spec R9; X8 |
| R5.9 | Micro-batched output is compared with batch-1 tokens and the agreement rate is reported. Batched runs are never used for exact-equality claims. | Should | W5 | B (unconfirmed) | Spec R9; G3 |
| R5.10 | Tune the KV admission budget from measured KV use. | Could | W6 | A (unconfirmed) | Proposal PG3 |
| R5.11 | Decision D7: batch prefill as well as decode? Default is decode only. Status: decided 2026-11-07 (#41): decode only; the cost is that a prefill stalls batched decodes, which X8 reports. | Could | W5 | A and B (unconfirmed) | D7 |
| R5.12 | Decision N4: the step message carries an absolute deadline in unix milliseconds, but the two laptops' wall clocks may differ. Choose either a clock-offset check at start or gateway-only deadline enforcement. Blocks R5.4. Status: decided 2026-11-07 (#32): gateway-only enforcement; each hop carries the time left as a relative budget (field 9 retired, budget_ms added, #36). | Should | W3 | A and B (unconfirmed) | G4; protocol |

### Epic E6. Failure recovery and fencing

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R6.1 | Fault injector with kill (SIGKILL on macOS, TerminateProcess on Windows), suspend and resume, and a per-step sleep, using psutil and invoked through the supervisors. Skeleton in W1, complete in W4. | Must | W1 | B (unconfirmed) | Proposal PG4; G1 |
| R6.2 | Supervisor on each laptop spawns its worker and respawns it automatically after exit, with no manual step; respawn time is logged. | Must | W4 | B (unconfirmed) | Spec R10 |
| R6.3 | Failure detection by stream break, heartbeat timeout or step deadline; detection time is logged with gateway timestamps. | Must | W4 | A (unconfirmed) | Spec R8; Spec R10 |
| R6.4 | On detecting a dead stage, the gateway declares its epoch dead, waits for the new Register, assigns a new epoch and sends a route update. | Must | W4 | A (unconfirmed) | Spec R10; Spec R12 |
| R6.5 | The gateway bumps the attempt for every request with KV state on the dead stage, including idle requests, not only in-flight ones. Test with an idle request holding KV. | Must | W4 | A (unconfirmed) | Spec R11; Proposal PG4 |
| R6.6 | Chunked replay: resend the prompt as one message (start_pos 0), then one single-token message per already-emitted token, with the original chunk boundaries and the replay flag. Stages truncate and recompute. Replay goes through both stages. | Must | W4 | A (unconfirmed) | Spec R11; R2.8 |
| R6.7 | Replay verification: recomputed tokens are compared with stored history and not re-emitted; a mismatch is logged as a divergence. The request completes with output equal to a fault-free run (batch 1). | Must | W4 | A (unconfirmed) | Spec R11; X5 |
| R6.8 | Stale-output test: suspend Stage 1 past the heartbeat timeout, let the replacement serve, resume the old process, and check every late output is rejected and the client sees zero duplicate tokens. Repeat for Stage 0. | Must | W4 | A (unconfirmed) | Spec R12; X6 |
| R6.9 | Each recovery increments the attempt counter; a request that cannot be recovered fails with an error at its deadline. | Must | W4 | A (unconfirmed) | Spec R8; Proposal PG4 |
| R6.10 | Slow worker: an injected per-step sleep beyond the step deadline fails the request cleanly at its deadline, and queue bounds still hold. | Must | W4 | A (unconfirmed) | Proposal PG4 |
| R6.11 | Delayed or lost hop: the transport wrapper can delay or drop a hop message; the request recovers by timeout and replay or fails at its deadline. | Should | W5 | A (unconfirmed) | Proposal PG4 |
| R6.12 | Recovery time is logged as four parts: detection, respawn, weight load, replay prefill, plus recomputed tokens and stale outputs discarded. | Must | W4 | A (unconfirmed) | X5; Proposal PG6 |
| R6.13 | Each failure-handling claim in this epic has an automated fault-injection test that runs on both laptops. | Must | W4 | A and B (unconfirmed) | G2 |
| R6.14 | Both stages failing at once recovers, with longer recovery time (test). | Could | W5 | A (unconfirmed) | Fault model |
| R6.15 | Stretch, partial replay (experiment X10): Stage 0 keeps its outbound activations per request so a Stage 1 failure replays only Stage 1; compare recompute time with X5. | Could | W6 | B (unconfirmed) | X10; D6 |
| R6.16 | Stretch, gateway durability: a durable request log lets the gateway restart. Default is out of the fault model, stated in the report. | Could | W6 | A (unconfirmed) | D5 |
| R6.17 | Decision D6: replay scope. Default is full replay through both stages; the alternative is partial replay (R6.15). Decide by W5. Status: decided 2026-11-07 (#51): full replay only; R6.15 stays a stretch, revisited only if Stage 0 busy time during replay in the X5 Stage 1 kill runs exceeds a quarter of recovery time and every Must exit criterion passes. | Should | W5 | A and B (unconfirmed) | D6 |
| R6.18 | Decision TA4: is full-prompt replay the expected recovery, or should the system checkpoint the KV cache? Status: decided 2026-11-01 (#43): full-prompt replay, as the specification states. | Should | W3 | A and B (unconfirmed) | TA4; Spec R11 |

### Epic E7. Measurement and experiments

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R7.1 | JSON-lines metrics logger: one line per request and one per hop message, timestamped at the gateway with one clock. | Must | W2 | A (unconfirmed) | G4; Proposal PG6 |
| R7.2 | Transport counters: serialized bytes per message type per hop, separating prefill from decode. | Must | W2 | A (unconfirmed) | Spec R16; X3 |
| R7.3 | One plotting script produces every plot and summary table from the run logs. Built as `pllm.report` (#56): tables and plots per run and a table across runs; each experiment adds its own plots across runs. | Must | W3 | A (unconfirmed) | G4; Proposal PG6 |
| R7.4 | Report TTFT, TPOT and end-to-end time per request at p50, p95 and p99. Defined as in vLLM's serving benchmark: TTFT from arrival at the gateway, TPOT excluding the first token, linear-interpolation percentiles (#56). | Must | W3 | A (unconfirmed) | Spec R14 |
| R7.5 | Report system-wide throughput in output tokens per second. | Must | W3 | A (unconfirmed) | Spec R15 |
| R7.6 | Report stage utilisation (busy time per stage). Each Beat carries the stage's running busy and step totals (schema 3), logged by the gateway on arrival (#56). | Should | W3 | A (unconfirmed) | X2 |
| R7.7 | Transport delay-injection wrapper with 0, 1, 5, 20 and 50 ms per message on the two LAN hops (gateway to Stage 0 and Stage 0 to Stage 1), injected on Host A and pipelined, not blocking; toxiproxy as an optional cross-check. | Must | W5 | A (unconfirmed) | X4; Proposal PG6 |
| R7.8 | X1a (single host, exact) and X1b (two laptops, agreement rate, first divergence position, maximum logit difference against each host's single-process reference). | Must | W2 | B (unconfirmed) | X1a; X1b; Spec R13 |
| R7.9 | X2: concurrency 1, 2, 4, 8, 16 against single-process round-robin serving without batching, the primary on Host A only, the reference across the two laptops at concurrency 1, and one Host B single-process run at concurrency 1 that records peak RSS; reports throughput, TTFT, TPOT and stage utilisation. | Must | W3 | A (unconfirmed) | X2; Spec R15 |
| R7.10 | X3: prompt lengths 16, 128 and 512, 128 decoded tokens; measured bytes per token per hop equal hidden size x 4 B x tokens within framing overhead. | Must | W2 | B (unconfirmed) | X3; Spec R16 |
| R7.11 | X4: per-token latency and throughput against injected delay of 0, 1, 5, 20 and 50 ms per message on the two LAN hops, at concurrency 1 and at a higher concurrency. | Must | W5 | A (unconfirmed) | X4; Spec R14 |
| R7.12 | X5: kill Stage 1 and separately Stage 0 at token 20 of 64 with 4 requests in flight; detection, respawn, load and replay times; output equals reference. | Must | W4 | A (unconfirmed) | X5; Spec R10; Spec R11 |
| R7.13 | X6: suspend Stage 1 past the heartbeat timeout and resume after replacement; repeat for Stage 0; stale outputs by reason; duplicate tokens at the client. | Must | W4 | A (unconfirmed) | X6; Spec R12 |
| R7.14 | X7: offered load above capacity with admission bound on and off, and the credit limit varied; queue depth over time, rejects, p99 latency. With bounds, depth is capped and p99 is bounded. | Must | W3 | A (unconfirmed) | X7; Spec R7 |
| R7.15 | X8: 16 concurrent requests, B in {1, 2, 4, 8}, W in {0, 2, 5} ms; throughput, latency (TPOT p95 and p99 included, to show the stall a prefill causes) and token agreement against batch 1. | Should | W5 | B (unconfirmed) | X8; Spec R9 |
| R7.16 | X9: layer split from even down to memory- and speed-balanced; stage busy time, tokens per second, Host B peak RSS and page faults. | Must | W5 | B (unconfirmed) | X9; Proposal PG3 |
| R7.17 | Every LAN experiment has at least 5 repeats and reports the spread; each run records the network and round-trip time. | Must | W6 | A and B (unconfirmed) | Proposal evaluation setup |
| R7.18 | Reported numbers come only from the scripts in R7.3; run outputs are not committed. | Must | W6 | A and B (unconfirmed) | G4; section 7.3 |
| R7.19 | Decision N5: X2 comparison points. Primary is single-process round-robin without batching. Reference is the pipeline at concurrency 1 (gain at most about 2x). Upper bound is a batched single process on Host A, which may beat the pipeline; it is not yet in the X2 plan. Decide whether to add it. Status: decided 2026-11-07 (#57), option 3: primary and reference now, the upper bound after R5.7 (#40), the primary on Host A only plus one Host B single-process run at concurrency 1; if micro-batching is cut, the upper bound is cut and the report says so. | Should | W3 | A and B (unconfirmed) | Proposal expected results; X2 |
| R7.20 | Decision N6: with the gateway and Stage 1 on Host A, only two hops cross the LAN, so a per-hop delay gives about 2 x delay per token, not 3 x. Confirm which hops get injected delay before fixing the X4 expectation. Status: decided 2026-11-07 (#48): the two LAN hops get the delay, and the X4 expectation is about 2 x delay per token. | Should | W5 | A and B (unconfirmed) | Proposal expected results; X4 |
| R7.21 | Decision N7: an even split of Qwen3-0.6B may also fit in Host B's budget, so X9 may not separate the splits on memory. Keep Qwen3-0.6B and report speed as well as memory, or add Qwen3-1.7B where memory binds. Status: decided 2026-11-07 (#61): SmolLM2-135M for all runs for now, so X9 runs on SmolLM2-135M, where an even split fits Host B and the splits separate on speed and busy time, not on memory; Qwen3-0.6B stays pinned and tested, and Qwen3-1.7B stays a stretch (R8.12). | Should | W5 | A and B (unconfirmed) | X9; Proposal expected results |

### Epic E8. LAN deployment on both laptops

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R8.1 | Stage 0 on Host B and Stage 1 plus gateway on Host A run as separate processes over the LAN and one request completes end to end. | Must | W2 | A and B (unconfirmed) | Spec R1; Spec R2; D9 |
| R8.2 | Launch scripts written in Python start each process the same way on macOS and Windows. | Must | W2 | A and B (unconfirmed) | G7; Spec R2 |
| R8.3 | Windows setup documented and tested: native Python (not WSL2), worker port reachable (for now Public profile with the firewall off, agreed 2026-11-09), charger and performance power plan. | Must | W2 | B (unconfirmed) | Proposal special requirements |
| R8.4 | macOS setup documented and tested: incoming connections allowed, charger connected. | Must | W2 | A (unconfirmed) | Proposal special requirements |
| R8.5 | Network fallbacks (direct cable, phone hotspot, Tailscale) work by changing the configuration file only. | Should | W2 | A (unconfirmed) | Proposal special requirements |
| R8.6 | Capacity-aware layer split: a function chooses k from each host's measured free memory and speed; Host B receives fewer layers than an even split. | Must | W5 | A (unconfirmed) | Spec R3; Proposal PG3 |
| R8.7 | Host B peak RSS for the chosen split stays within the stated budget and shows no paging to disk (page-fault counter reported). | Must | W5 | B (unconfirmed) | X9; G5 |
| R8.8 | Presented runs use SmolLM2-135M fp32 with the balanced split until the pipeline, recovery and the main experiments work end to end (decided 2026-11-07, #61); they switch to Qwen3-0.6B if time and the Host B budget allow. SmolLM2-135M is also used for development and tests. | Must | W5 | A and B (unconfirmed) | D1; Proposal goals |
| R8.9 | Fixed thread count per process, identical in reference and workers for X1a; on a single host each stage gets half the cores. | Must | W2 | B (unconfirmed) | X1a; G3 |
| R8.10 | Single-host mode: all processes start on one laptop with the same entry points, for tests and X1a. | Must | W2 | A (unconfirmed) | Proposal special requirements |
| R8.11 | Run discipline: other programs closed, same power state for every run, available memory logged at the start of each run. | Should | W2 | A and B (unconfirmed) | Proposal evaluation setup |
| R8.12 | Stretch: Qwen3-1.7B with the stage role swap and about 6 layers on Host B, only if measured Host B memory allows. | Could | W6 | B (unconfirmed) | D1; Proposal goals |
| R8.13 | Decision D12: role of the spare Linux MacBook (gateway and dashboard host, warm spare worker, or unused). Pending since before W2. | Could | W2 | A and B (unconfirmed) | D12 |

### Epic E9. Dashboard, discovery and pairing (stretch)

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R9.1 | Decision D11: keep the dashboard, discovery and pairing scheduled in W3 to W5 or treat them as stretch behind the minimum scope (pipeline, fencing, replay, tests, experiments). The scope proposal lists them as stretch. Owners are unassigned (about 2 days each). Blocks all of Epic E9. | Must | W3 | A and B (unassigned) | D11; Proposal deliverables |
| R9.2 | Dashboard: one HTML page with plain JavaScript, served next to the gateway by a small HTTP endpoint, live updates over server-sent events. Built as `pllm.dashboard`, started with `--dashboard` (#68). | Could | W3 | Unassigned | D11 |
| R9.3 | Input controls: prompt box, send, send 8 at once, flood; live token stream per request. Each request goes through the gateway's Frontend, so refusals and deadlines are real (#68). | Could | W3 | Unassigned | D11 |
| R9.4 | Status and counters: each stage alive or dead (or unknown, before it registers or when its Beats are late but it is not yet declared dead), epoch, queue length, free RAM; tokens per second, rejects, stale outputs discarded, duplicate tokens (always 0). Free RAM comes from each laptop's supervisor; duplicates are counted on the streams the page relays (#68). | Could | W4 | Unassigned | D11 |
| R9.5 | Fault controls: kill Stage 0, freeze, unfreeze, calling the supervisor on that laptop. For both stages, through the dashboard server, since the supervisor refuses cross-origin browser calls (#68). | Could | W4 | Unassigned | D11 |
| R9.6 | Experiments run from scripts; no reported number depends on a browser. | Must | W6 | A and B (unconfirmed) | G4 |
| R9.7 | mDNS announce on service `_pllm._tcp.local` with node id, OS, free RAM, cores, app version and weights hash; a scan lists the nodes found. | Could | W5 | Unassigned | D11 |
| R9.8 | Add-by-IP is always available and the demo never depends on discovery working. | Could | W5 | Unassigned | D11; R1.8 |
| R9.9 | Pairing: the selected node shows a 6-digit code; a shared session token is derived and carried in RPC metadata; a missing or wrong token is rejected with UNAUTHENTICATED (test). | Could | W5 | Unassigned | D11 |
| R9.10 | Coordinator assigns roles and the layer split from announced free RAM, feeding R8.6. | Could | W5 | Unassigned | D11; X9 |
| R9.11 | The report states that pairing is a LAN-trust measure and not strong security. | Could | W6 | Unassigned | D11; Fault model |

### Epic E10. Report, demo and submission

| ID | Requirement | Priority | Week | Owner | Source |
|---|---|---|---|---|---|
| R10.1 | Decision N8: confirm the real submission and demo dates with the course. The plan assumed about 2026-11-25. Status: the course has not released the dates; the probable submission deadline is 2026-11-15 and the demo is probably around 2026-11-20 (reported by Member B on 2026-11-07, not confirmed by the course, #70). Blocks re-baselining section 6. | Must | W3 | A and B (unconfirmed) | Roadmap assumption |
| R10.2 | Final report: description, architecture, protocol, guarantees, results with plots, limitations. Each member writes the sections for their own components and experiments; introduction and conclusion are joint. | Must | W6 | A and B (unconfirmed) | Proposal deliverables |
| R10.3 | A contribution table in the report that matches the git history and this document's owners. | Must | W6 | A and B (unconfirmed) | Contribution rule |
| R10.4 | Evidence table: each of Spec R1 to Spec R16 cites a test, a run or a plot. | Must | W6 | A and B (unconfirmed) | O9 |
| R10.5 | Limitations stated up front: single-request latency is higher than single-process; gateway is a single point of failure; computation is at-least-once while delivery is exactly-once; cross-architecture results are agreement, not equality; the batched single-process upper bound may win. | Must | W6 | A and B (unconfirmed) | Proposal expected results |
| R10.6 | Demo script with segments split evenly: placement and memory; correctness; concurrency and backpressure; crash recovery; fencing; results. Each member presents what they built. Rehearsed twice. | Must | W6 | A and B (unconfirmed) | Contribution plan |
| R10.7 | Demo runs live on both laptops including a kill and a suspend; a recorded run exists as a fallback if the LAN fails on the day. | Should | W6 | A and B (unconfirmed) | Proposal special requirements |
| R10.8 | Submission package verified from a clean checkout on both OSes: lockfile sync, tests, one end-to-end run. | Must | W6 | A and B (unconfirmed) | G6; G7 |
| R10.9 | Buffer days reserved for fixes only; stretch work starts only if every Must exit criterion has passed. | Should | Buffer | A and B (unconfirmed) | Roadmap |

## 9. Risks and open questions

### Risks

| ID | Risk | Mitigation |
|---|---|---|
| K1 | Schedule: two weeks behind, six weeks of planned work in four weeks plus 3 buffer days (section 6). | Recovery note in section 6; stretch cut first; flag slips the day they are seen. |
| K2 | Host B memory is the binding constraint (about 2.5 GB assumed). Paging makes latency spiky and ruins timings. | Measure in W1 (R1.7); memory-mapped embedding; capacity-aware split; admission by KV budget; report page faults. |
| K3 | ARM and x86 fp32 kernels differ in low bits, so two-laptop tokens can diverge after a near-tie. | Claim exactness per host only; report agreement, first divergence and maximum logit difference. |
| K4 | A single request is slower than single-process (two LAN hops per token); throughput gain depends on concurrency. | State the expectation before results; report the pipeline at concurrency 1 as reference. |
| K5 | The campus network may block host-to-host traffic or multicast. | Network check on day one of W2 (R1.8); fallbacks by configuration only; Add-by-IP default. |
| K6 | Windows specifics: firewall prompts, power plan, background load. | Native Python, documented firewall rule, run discipline (R8.3, R8.11). |
| K7 | Replay bit-exactness depends on reusing the original chunk shapes. | Chunked replay (R6.6); W1 chunking test (R2.8). |
| K8 | The own Llama forward slips past W1. | Fallback GPT-2 (N1); distributed-systems requirements are model independent. |
| K9 | Gateway failure is not tolerated. | Stated as out of scope; durable log is a stretch (R6.16). |
| K10 | Apple Silicon mixes performance and efficiency cores, which adds timing noise. | Pinned thread count; at least 5 repeats; report spread. |
| K11 | Clock skew between laptops affects absolute deadlines. | N4 (R5.12), decided (#32): only the gateway judges deadlines and hops carry a relative budget; latencies measured only at the gateway. |
| K12 | Dashboard and pairing work competes with core work for the same two members. | D11 decision (R9.1); stretch only after Must exit criteria pass. |

### Open decisions and questions

Items reported by one member and not yet confirmed by the other are marked unconfirmed. Decided items are listed for completeness.

| Id | Decision or question | Default or status | Blocks |
|---|---|---|---|
| D1 | Model: SmolLM2-135M for development, Qwen3-0.6B for headline runs, with a capacity-aware split. Wording differs across project documents (headline versus headline stretch). | Decided 2026-11-03: as stated, one wording (#9). Updated 2026-11-07 (#61): SmolLM2-135M for all runs for now | R1.12, R8.8 |
| D4 | Transport: gRPC with raw tensor bytes against ZeroMQ. | Decided 2026-11-01: gRPC with raw tensor bytes (#4) | R1.4 |
| D5 | Gateway durability. | Default: out of the fault model; revisit in W6 | R6.16 |
| D6 | Replay scope: full replay or partial replay (X10). | Default full. Decided 2026-11-07 (#51): full replay | R6.15, R6.17 |
| D7 | Micro-batching depth: decode only or prefill too. | Default decode only. Decided 2026-11-07 (#41): decode only | R5.11 |
| D11 | Dashboard, discovery and pairing: scheduled W3 to W5 or stretch. | Unconfirmed; owners unassigned | R9.1 |
| D12 | Role of the spare Linux MacBook. | Pending since before W2 | R8.13 |
| D2, D3, D8, D9, D10 | Deployment (gateway and Stage 1 on macOS, Stage 0 on Windows); CPU fp32; hardware confirmed; Host B runs Stage 0; even split of work. | Decided. D10 owners are still marked unconfirmed in this document. | none |
| TA1 | Is the model ours to choose or supplied? | Decided 2026-11-01: ours (#8) | R1.11 |
| TA2 | Is a gateway process acceptable alongside the two workers? | Decided 2026-11-01: separate gateway process (#10) | R1.13 |
| TA3 | Is bit-exact equality per host plus cross-host agreement acceptable? | Decided 2026-11-01: exact per host, agreement across hosts (#11) | R1.14 |
| TA4 | Is full-prompt replay the expected recovery, or KV checkpointing? | Decided 2026-11-01: full-prompt replay (#43) | R6.18 |
| N1 | Fallback to GPT-2 if the own forward slips. | Not needed 2026-11-01: own forward matches (#13) | R2.11 |
| N2 | Ordering stream wording: bidirectional (specification mapping) or client-streaming (protocol). | Decided 2026-11-01: client-streaming as in the scope proposal (#19) | R3.16 |
| N3 | gRPC 4 MiB message limit against long prefills. | Decided 2026-11-03: receive limit from config, no chunking (#20) | R3.17 |
| N4 | Absolute deadlines and unsynchronised clocks. | Raised in this document. Decided 2026-11-07 (#32): gateway-only enforcement with a relative budget on each hop | R5.12 |
| N5 | X2 comparison points, adding the batched single-process upper bound. | Raised in this document. Decided 2026-11-07 (#57): option 3 | R7.19 |
| N6 | X4 delay hops: expect about 2 x delay or 3 x delay per token. | Raised in this document. Decided 2026-11-07 (#48): the two LAN hops | R7.20 |
| N7 | X9 framing if an even split also fits Host B. | Raised in this document. Decided 2026-11-07 (#61): SmolLM2-135M for all runs for now | R7.21 |
| N8 | Real submission and demo dates. | Raised in this document | R10.1 |

## 10. Traceability

### Course specification clauses

| Clause | Summary | Requirements |
|---|---|---|
| Spec R1 | Small model, two worker processes | R1.11, R1.13, R2.1, R3.1, R8.1 |
| Spec R2 | CPUs or GPUs | R2.10, R3.15, R8.1, R8.2 |
| Spec R3 | Split layers between workers | R2.5, R2.9, R3.1, R4.2, R8.6 |
| Spec R4 | Transfer activations over RPC | R1.4, R3.4 |
| Spec R5 | Per-request ordering | R3.7, R3.16, R4.2, R4.5 |
| Spec R6 | KV-cache state | R2.7, R3.5, R3.6, R3.8, R3.9, R4.6 |
| Spec R7 | Bounded request queues | R5.1, R5.2, R5.3, R7.14 |
| Spec R8 | Timeouts | R3.10, R5.4, R5.5, R5.6, R6.3, R6.9 |
| Spec R9 | Optional micro-batching | R5.7, R5.8, R5.9, R7.15 |
| Spec R10 | Restart the affected worker | R3.3, R3.10, R3.13, R4.4, R6.2, R6.3, R6.4, R7.12 |
| Spec R11 | Replay the prompt to rebuild state | R6.5, R6.6, R6.7, R6.18, R7.12 |
| Spec R12 | Discard stale outputs by request and step ids | R3.11, R3.12, R4.7, R4.8, R6.4, R6.8, R7.13 |
| Spec R13 | Compare with single-process inference | R1.14, R2.3, R2.4, R2.9, R7.8 |
| Spec R14 | Report token latency | R7.4, R7.11 |
| Spec R15 | Report throughput | R7.5, R7.9 |
| Spec R16 | Report transferred bytes | R7.2, R7.10 |

### Scope proposal goals

| Goal | Requirements |
|---|---|
| PG1 Pipeline model parallelism | R1.1, R1.4, R2.1, R3.4, R4.1, R8.1 |
| PG2 Correct ordering and state | R2.7, R3.5, R3.6, R3.7, R3.8, R4.3, R4.5 |
| PG3 Backpressure and flow control | R5.1, R5.2, R5.3, R5.4, R5.10, R7.16, R8.6 |
| PG4 Failure recovery | R2.8, R3.3, R3.11, R4.3, R4.7, R6.1, R6.5, R6.6, R6.9, R6.10, R6.11 |
| PG5 Correctness validation | R1.9, R2.3, R2.9, R7.8 |
| PG6 Measurement | R6.12, R7.1, R7.3, R7.7, R7.17 |

### Epics to objectives

| Epic | Objectives |
|---|---|
| E1 Foundations and environment | O1, O9 |
| E2 Reference model and equality oracle | O2 |
| E3 Stage workers and protocol | O1, O3, O5 |
| E4 Gateway and decode loop | O1, O3, O5 |
| E5 Flow control and micro-batching | O4, O7 |
| E6 Failure recovery and fencing | O5 |
| E7 Measurement and experiments | O2, O3, O4, O5, O6, O7, O8 |
| E8 LAN deployment on both laptops | O1, O8 |
| E9 Dashboard, discovery and pairing | O10 |
| E10 Report, demo and submission | O9 |

## 11. Change log

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-10-26 | First draft from the scope proposal, the course specification and the team's design and plan. For review by both members. |
| 1.1 | 2026-11-01 | Decisions D4, TA1 to TA4, N1 and N2 recorded as decided, with their issues. |
| 1.2 | 2026-11-03 | Decisions N3 (#20) and D1 (#9) recorded as decided. |
