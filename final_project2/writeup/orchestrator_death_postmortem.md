# Orchestrator death post-mortem — root cause: SIGHUP from Claude Code shell exit

## Symptoms

Phase 2A orchestrator (`scripts/_run_phase2_after_2b.sh`, launched via Claude
Code Bash tool with `run_in_background: true` at 2026-05-09 ~05:14 EDT) died
~13 minutes in, with the following evidence:

- `logs/post2b_MASTER.log` (master orchestrator stdout): contains only the
  initial header `===== Phase 2A start =====` (27 bytes, mtime 05:15:04).
  Never wrote `[phase2a] elapsed Ns` (line 21 of the script), so the
  `bash scripts/_run_phase2a.sh` invocation on line 20 never returned.
- `logs/phase2a_MASTER.log` (inner Phase 2A stdout): completed batches 1
  and 2, dispatched batch 3, then stopped (mtime 05:25). No batch-3
  `[done N]` lines, no `[batch3] elapsed Ns`.
- Inner training logs in `logs/phase2a/`:
  - Batch 1+2 runs (bc×3, td3_bc×3, cql_vanilla×2): completed cleanly
    with metrics.json written.
  - Batch 3 runs that were dispatched (cql_vanilla_seed2024,
    awac×3): tqdm progress writes stop at ~01:28 elapsed (e.g.,
    awac_seed42 at iteration 5464/20000 = 27%; cql_vanilla_seed2024 at
    3270/20000 = 16%). No Python traceback, no OOM kill, no
    `KeyboardInterrupt`, no shutdown message.

The four batch-3 children all halt at the same elapsed wall-clock with
no error output. That is the SIGHUP cascade signature.

## Root cause

Claude Code's `run_in_background: true` flag for the Bash tool does **not**
detach the launched process from the shell session. The child inherits
the controlling terminal and process group from Claude's shell wrapper.
When Claude's conversation ends and the wrapper shell exits (which it
did at ~05:27 EDT — 14 hours before this session resumed), the kernel
sends SIGHUP to every process group attached to the closing terminal.
The orchestrator and all 4 batch-3 training children received SIGHUP
simultaneously and exited.

Cross-check on the time alignment:
- Master started 05:15:04, batch 1 wall-clock 182s, batch 2 wall-clock
  450s. Batch 3 dispatched at 05:15+182+450 = 05:25:36.
- Children killed 84-88s into batch 3 → 05:27:00 ± 0:01.
- This matches the Claude Code session termination time exactly.

The previous session's note "backgrounded children survive even when
wrapping shell exits" was wrong for this environment. The
`run_in_background` flag and bare `&` both produce children that share
the controlling tty with the parent and die with it.

## What got lost

- Phase 2A: 16 of 24 runs missing.
  - Completed (8/24): bc×3, td3_bc×3, cql_vanilla×2 (seeds 42, 1337).
  - Killed mid-training: cql_vanilla_seed2024 (16%), awac×3 (~27%).
  - Never started: bcq×3, iql×9 (the λ-anchor).
- Phase 2C: 0/6 — orchestrator never reached the post-2A scheduling block.
- Phase 2D: 0/3 — same.
- Aggregation + plot: never ran.

Phase 2B (9/9) is intact because it ran before the orchestrator started.

## The fix

Wrap the orchestrator in a session-surviving manager so it's owned by a
process tree independent of Claude's shell. In order of preference:

```bash
# 1. tmux (preferred — survives terminal close, easy to re-attach for
#    inspection, standard on the cluster)
tmux new-session -d -s phase2 'bash scripts/_run_phase2_after_2b.sh'
tmux ls                                # verify session created
tail -f logs/post2b_MASTER.log         # 30s confirm log writes

# 2. screen (fallback if tmux unavailable)
screen -dmS phase2 bash scripts/_run_phase2_after_2b.sh

# 3. setsid + nohup + disown (last resort; brittle on cluster scheduler)
setsid nohup bash scripts/_run_phase2_after_2b.sh \
    > logs/post2b_MASTER.log 2>&1 < /dev/null &
disown
```

Verification before walking away (mandatory):
1. `tmux ls` shows the `phase2` session.
2. `tail -f logs/post2b_MASTER.log` shows fresh writes for ~30 seconds.
3. `pgrep -f run_phase2_after_2b` returns a PID owned by the user, with
   `ps -o stat= -p <PID>` not in `T` / `Z` state.

## What does NOT need to change

- The script logic itself is correct. Once protected from SIGHUP, the
  same orchestrator would run to completion. (Idempotency under partial
  re-run is a separate concern — see `pre-flight 3` notes.)
- Phase 2B is unaffected (already complete and on disk).
- The diagnosis does not implicate any Python / training code; the
  killed runs were healthy mid-stride.

## Decision

Re-launch via `tmux new-session -d -s phase2 ...` after Pre-flight 3
clears (idempotency). Do not re-use `run_in_background: true` for jobs
that need to outlive the conversation.
