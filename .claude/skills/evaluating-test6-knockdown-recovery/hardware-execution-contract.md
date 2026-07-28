# Hardware-adaptive completion contract

Apply this contract after the user confirms the resolved statistical configuration.

## Invariants

Prioritize completion first and speed second. Never skip a step or reduce/change FDR,
LFC, seed, repeat count, minimum cells, gene/guide scope, feature caps, shuffle mode,
or any other statistical parameter to improve runtime. Only tune CPU threads, the
non-parametric execution engine, and worker counts.

Measure and print CPU affinity, total/available RAM, and GPU count/model/free VRAM
before selecting a tier. Keep `workers × threads` at or below CPU affinity. Treat RSC
as one serialized GPU process; parallelize only independent CPU PyDESeq2 work around it.
Do not run two full-data skills concurrently on the same host unless their combined
measured memory plans fit with at least 25% RAM left free.

## De-escalation ladder

Start at the fastest compatible tier and only move downward:

0. RSC, one GPU process, one thread for the GPU leg, hardware-capped CPU workers for
   PyDESeq2.
1. Same engine and threads, half the Tier-0 CPU workers.
2. Same engine, one worker, CPU-affinity threads for CPU work.
3. Arc pdex on CPU with hardware-capped moderate workers.
4. Arc pdex, one worker, one thread.

If RSC was not selected or no compatible GPU exists, begin at Tier 3. Never silently
substitute pdex for a user-confirmed RSC-only comparison: record the tier transition
and engine change in the run report.

## Failure and stuck detection

Run each attempt through the bundled `run_with_watchdog.py`. Define progress as either
new stdout/stderr bytes or a new/growing artifact below the step output directory.
A step is stuck after 20 minutes without either signal. Also impose a preflight-confirmed
wall-clock budget for each step. On a non-zero exit, exit 137, `MemoryError`, CUDA OOM,
OOM-kill evidence, idle timeout, or wall timeout:

1. terminate the entire attempt process group;
2. preserve completed caches/checkpoints;
3. reduce exactly one tier;
4. retry the same step with identical statistical arguments.

For OOM, select a next tier with a strictly lower estimated peak-memory plan. Never
wait indefinitely for a quiet process.

Overview and Test 6 use resumable per-target PyDESeq2 checkpoints automatically.
Test 1 must use `--signature-cache-dir` with `--resume-signatures`. Reuse any other
strictly matching cache supported by the step.

If Tier 4 fails, stop the campaign and report the exact command, tier history, trigger,
exit code, last progress time, and expected artifacts. Do not mark the step complete
until every expected artifact exists and is non-empty.

## Required report

For every step report the chosen tier, exact command, wall time, peak process-group RSS,
hardware measurements, artifact paths, and every de-escalation trigger. Put any change
outside the permitted performance knobs at the top of the report.
