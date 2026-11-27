# Live demo script

A 15 minute live demo on both laptops: placement and memory, correctness, concurrency and
backpressure, crash recovery, fencing and results. Each member presents what they built.

Status: the presenter split is proposed, to be agreed by both members. It follows the git
history of each component.

- Member A runs the Mac (Host A): the gateway, Stage 1, the dashboard and the projector.
- Member B runs the Windows laptop (Host B): Stage 0.
- Model: SmolLM2-135M for every run.
- Network: the college Wi-Fi, with the phone hotspot as fallback (see "Network" in the
  README).

## Timing

| # | Segment | Presenter (minutes) | Time |
|---|---|---|---|
| 0 | Opening: what runs where | A 1 | 1 |
| 1 | Placement and memory | B 1.5, A 0.5 | 2 |
| 2 | Correctness | A 1, B 1 | 2 |
| 3 | Concurrency and backpressure | B 1.5, A 0.5 | 2 |
| 4 | Crash recovery (live kill) | A 2, B 0.5 | 2.5 |
| 5 | Fencing (live suspend) | B 2, A 0.5 | 2.5 |
| 6 | Results | A 1.5, B 1.5 | 3 |
| | Total | A 7, B 8 | 15 |

## Before the day

- Both laptops: `uv sync` and `uv run python -m pllm.models smollm2-135m`, on the same commit.
- The experiment runs are done and their files are in `runs/` on the Mac: X1a on each
  laptop, X1b, X2, X3, X4, X5, X6 and X7. Segment 6 shows them; nothing in it is run live.
- The report is built: `uv run python -m pllm.report`, output in `runs/report`. The X7 table
  is built with `uv run python -m pllm.x7 report <x7 run files> --out runs/x7-report`.
- One full rehearsal is recorded as the fallback (see [Recorded fallback](#recorded-fallback)).
- For the segment 5 fallback, know the Mac's thread count for Stage 1.

## Setup (about 10 minutes, before the audience)

1. Follow the README "Session checklist", steps 1 to 9, in order. Step 9 gives the round
   trip in ms; note it for the fallback commands (`--rtt-ms`). Steps 5 and 6 below replace
   its step 10: start the launchers only once, with the demo's options.
2. In `config/hosts.local.toml` on both laptops, set `[supervisor0]` to the Windows laptop's
   address and `[supervisor1]` to the Mac's (or delete both tables). Otherwise Kill, Freeze,
   Unfreeze and the memory panel fail for Stage 0.
3. Keep `[dashboard]` at `127.0.0.1`. The dashboard and `pllm.client` run on the Mac only:
   their channels to the gateway have no keepalive, and the dashboard has no authentication.
4. Windows firewall (college Wi-Fi): the inbound rule under "Windows setup" in the README for
   TCP 50051 and 50061, limited to the Mac's address.
5. Start Stage 0 on Windows:

   ```
   uv run python -m pllm.launch smollm2-135m --roles stage0 --threads <Windows n>
   ```

6. Start the gateway, Stage 1 and the dashboard on the Mac:

   ```
   uv run python -m pllm.launch smollm2-135m --roles gateway stage1 --dashboard --heartbeat-timeout 5 --threads <Mac n>
   ```

7. Open `http://127.0.0.1:50080/` on the Mac. Both stages show alive with an epoch, and free
   memory shows for both laptops. If a laptop shows "supervisor unreachable", recheck steps 2
   and 4.
8. Send one warm-up prompt with Send.
9. Open a second Mac terminal for `pllm.client`, and keep the Windows launcher terminal
   visible (on its own screen or shared). Member B presents from the Mac screen.
10. Start the screen recording (see [Recorded fallback](#recorded-fallback)), so the live run
    is also recorded.

## Segments

### 0. Opening: what runs where (Member A, 1 minute)

Shown: the README's first section. The model is cut in two after layer k - 1 (k = 15 from
`[split]` in `config/hosts.toml`). Stage 0 runs on Windows, Stage 1 and the gateway on the
Mac. The gateway admits requests, orders decode steps and streams tokens. Stages exchange
activations over gRPC.

Audience sees: the dashboard with both stages alive.

Fallback: none needed.

### 1. Placement and memory (Member B 1.5, Member A 0.5; 2 minutes)

Member B shows:

- Stage 0 loads only its layer range, and reads embedding rows through the checkpoint's
  memory map, so only the rows of the tokens it gets enter memory.
- The Stage 0 worker's log in the Windows launcher terminal: its load time, then
  `serving, rss ... MiB`.
- The worker's memory against the 0.7 GiB budget of Host B, from the Windows supervisor (run
  on the Mac):

  ```
  curl http://<Windows IP>:50061/status
  ```

  `worker_rss_bytes` now, `worker_peak_rss_bytes` the lifetime peak (Windows reports it),
  `free_ram_bytes` the laptop's free memory.
- The free memory of both laptops on the dashboard's Stages panel.

Member A shows: the gateway's `KV admission: budget ...` line from its start, in the Mac
launcher terminal. For SmolLM2 at k = 15 it is 254.2 MiB, 23,040 bytes per position, 11,570
positions. Each request reserves its
prompt plus `max_tokens` positions before it may queue.

Audience sees: Stage 0's RSS beside the 0.7 GiB budget, and the admission rule that bounds
the KV cache inside it.

Fallback: the same lines in the recorded run's launch and supervisor logs.

### 2. Correctness (Member A 1, Member B 1; 2 minutes)

Member A runs the same prompt through the pipeline and through the single-process reference,
both on the Mac:

```
uv run python -m pllm.client "The capital of France is" --max-tokens 16 --timeout 30
uv run python -m pllm.reference smollm2-135m --text "The capital of France is" --new-tokens 16 --threads <Mac n>
```

Audience sees: the same tokens. Across two laptops the claim is agreement of tokens, not
bitwise equality. Then Member A shows the X1b summary: sequence and token agreement of the
pipeline against each host's reference.

Member B shows the X1a summary line: the split against the single process at split points 1,
n/4, n/2, 3n/4 and n-1, on one host, fp32, batch 1. It exits with 0 only at 100 percent token
match.

Fallback: the stored X1a and X1b result files in `runs/`. They need no network.

### 3. Concurrency and backpressure (Member B 1.5, Member A 0.5; 2 minutes)

Member B, on the dashboard:

1. Set tokens to 64 and press Send 8 at once. The 8 cards stream together; each request's
   tokens stay in order.
2. Press Flood (40). The gateway runs 14 requests, queues 8 and refuses the rest at once with
   RESOURCE_EXHAUSTED.

Audience sees: requests running at most 14, queued at most 8, steps in flight at most 8 (the
credits), the "Rejected" tile rising, and no request hanging.

Member A: KV admission reserves memory for each request before it queues. At these sizes it
does not bind; it binds with long requests (for example 6 requests of 2,000 positions: the
budget holds 11,570 positions, so the 6th is refused).

Fallback: the X7 table and `queue_depth.png` in `runs/x7-report`.

### 4. Crash recovery, live kill (Member A 2, Member B 0.5; 2.5 minutes)

A dashboard request of 64 tokens may end in about 4 to 8 s on the college Wi-Fi (a guess
from the median TPOT of two earlier gateway metrics files there: 66.8 ms for 1 request,
116.3 ms for 360 under load; check it at the first rehearsal). So start a long request first,
so that the kill lands while it runs. In the second Mac terminal:

```
uv run python -m pllm.client "<a long prompt>" --max-tokens 200 --timeout 120
```

Then press Send 8 at once, and Member B presses Kill on Stage 0.

Member B: the supervisor on Windows sees the worker exit and starts a new one with no manual
step.

Member A: the gateway sees the heartbeat stream break and declares the epoch dead (the
`detected` line in the Mac launcher terminal). Every running request gets a new attempt. When
the new Stage 0 registers with a higher epoch (the `registered` line), each request replays
from step 0 and is checked against the tokens already sent. The gateway writes one `recovery`
line per request, with the time in four parts, to its metrics file, not the terminal;
segment 6 shows them (`recovery.csv` in the report).

Audience sees: Stage 0 dead, then alive with a higher epoch. The client stream pauses and
continues with no repeated or missing token. The "Duplicate tokens (always 0)" tile stays 0.

Fallback: the recorded run. Backup, during questions only: run X5 once on Stage 0 against
the live system. It first runs its own fault-free reference, so it takes a few minutes:

```
uv run python -m pllm.x5 smollm2-135m --metrics runs/<gateway>.metrics.jsonl --stages 0 --repeats 1 --trial --rtt-ms <ms>
```

### 5. Fencing, live suspend (Member B 2, Member A 0.5; 2.5 minutes)

Start a long request again, as in segment 4. Then Member B presses Freeze on Stage 0.

Member A: Stage 0 still has a live process, so its supervisor does nothing. No Beat comes
for 5 s (`--heartbeat-timeout 5`), so the gateway declares the epoch dead and bumps every
running request's attempt. The requests wait for a new Stage 0.

Cue: Member B presses Unfreeze as soon as Stage 0 shows dead, before 8 s. After 8 s
keepalive closes the frozen worker's connections, and any outputs it held are lost instead
of arriving late.

Member B: the resumed process may send on a step it held. Its result carries the old attempt,
so the gateway's result filter drops it. The resumed process is then told it is dead at its
next Beat, exits, and its supervisor starts a new one. The new Stage 0 registers with a higher
epoch, and the requests replay and finish.

Audience sees: the "Stale outputs dropped" tile rises (its hover lists `stale_attempt`) when a
late output arrives, "Duplicate tokens (always 0)" stays 0, and the request ends with its full
output.

What this shows, stated as it is: live, the late outputs are rejected by their attempt. A
late output is not certain: a frozen worker that held no step sends none. The epoch checks
(`stale_epoch` at the gateway, `BAD_SENDER` at Stage 1) are shown by X6 in segment 6. No
Stage 0 replacement is started live, since two Stage 0 workers need about 1.4 GiB on Host B.

Fallback: the recorded run. Backup, during questions only, on the Mac:

- The test that proves fencing once a replacement serves, with a small random model. There
  Stage 1 refuses the late Stage 0 step with `BAD_SENDER`; live, the gateway drops it as
  `stale_attempt`:

  ```
  uv run pytest tests/test_stale_after_suspend.py
  ```

- Or X6 on Stage 1, with a Stage 1 standby started on the Mac before the demo:

  ```
  uv run python -m pllm.standby smollm2-135m --stage 1 --threads <Mac n>
  uv run python -m pllm.x6 smollm2-135m --metrics runs/<gateway>.metrics.jsonl --stages 1 --repeats 1 --trial --rtt-ms <ms>
  ```

### 6. Results (Member A 1.5, Member B 1.5; 3 minutes)

Every number shown comes from the scripts that read the metrics files, never from the
dashboard. Open `runs/report/index.md` and each run's plots.

Member A: X2 throughput against concurrency, X3 bytes per hop, X4 latency against injected
hop delay (the TPOT slope against the expected 2.0), and X1b agreement.

Member B: X5 recovery times in four parts, X6 late outputs rejected by reason, X7 the bounds
on and off (`runs/x7-report/table.md`).

This split is proposed: it follows who built each runner, and it differs from the experiment
owners in the BRD in this folder ([brd.md](brd.md): A owns X5 to X7, B owns X1b and X3). The members settle it.

Fallback: the files are on the Mac; nothing needs the network.

## If the LAN fails

1. Switch to the recorded run.
2. Segments 2 and 6 run from local files.
3. For segments 3 to 5 live, run everything on the Mac with the committed hosts file. On the
   Mac, `unset PLLM_CONFIG`, then:

   ```
   uv run python -m pllm.launch smollm2-135m --dashboard --heartbeat-timeout 5
   ```

   Say that both stages are now on one host.

## Recorded fallback

Record one full run of segments 0 to 6 at a rehearsal, and record the live run too.

- Mac: Cmd+Shift+5, Record Entire Screen (dashboard and terminals).
- Windows: Win+Alt+R (Game Bar) or a Snipping Tool video of the Stage 0 launcher terminal.
- Keep the run's gateway metrics file, launch records and report with the recording.
- Recordings are large: never commit them. Keep a copy on both laptops and one shared copy.

## Rehearsals

| Date | Network | Presenters | Segment times (0 to 6) | What failed | Recorded |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
