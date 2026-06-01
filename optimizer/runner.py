"""Parent-side driver: run a candidate in an isolated subprocess.

Enforces a wall-clock timeout, kills the whole process group on timeout, and
maps every failure mode (timeout, crash, OOM, garbage output) to a failed
BotResult so the optimizer loop never dies.

Platform note: on macOS `resource.RLIMIT_AS` is not reliably enforced, so a
memory-hungry bot is contained by this wall-clock timeout rather than the
in-process memory cap. On Linux the rlimit trips first. Either way the result
is a failed BotResult — the timeout is the primary cross-platform guard.
"""

import os
import signal
import subprocess
import sys

from optimizer.harness import BotResult


def evaluate(bot_path: str, seeds, timeout: int = 60, mem_limit_mb: int = 1024,
             invalid_rate_max: float = 0.05) -> BotResult:
    seeds_csv = ",".join(str(s) for s in seeds)
    cmd = [sys.executable, "-m", "optimizer.run_bot",
           bot_path, seeds_csv, str(invalid_rate_max), str(mem_limit_mb)]
    # start_new_session=True puts the child in its own process group so we can
    # kill any grandchildren it spawns.
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired as e:
        # The run_bot process group may still be alive; best-effort kill.
        _kill_stragglers(e)
        return BotResult.failed(f"timeout after {timeout}s")
    except OSError as e:
        # Spawn itself failed (e.g. fork/exec failure, resource exhaustion).
        # evaluate() must never raise, so map it to a failed result too.
        return BotResult.failed(f"spawn failed: {type(e).__name__}: {e}")

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["non-zero exit"]
        return BotResult.failed(f"subprocess exit {proc.returncode}: {tail[0]}")

    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return BotResult.failed("no output from subprocess")
    try:
        return BotResult.from_json(lines[-1])
    except Exception as e:
        return BotResult.failed(f"unparseable result: {type(e).__name__}: {e}")


def _kill_stragglers(exc):
    """Best-effort: kill the timed-out child's process group, if we can find it."""
    pid = getattr(exc, "pid", None)
    if pid is None:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
