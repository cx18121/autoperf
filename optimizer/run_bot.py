"""Subprocess entry point: validate + score ONE bot under resource limits.

Usage: python -m optimizer.run_bot <bot_path> <seeds_csv> <invalid_rate_max> <mem_mb>
Prints the BotResult as JSON on the final stdout line. Always exits 0 (a failed
bot is reported via BotResult.status, not a non-zero exit) unless setup itself
fails.
"""

import sys

from optimizer.harness import score_bot, BotResult
from optimizer.validator import validate_code


def _apply_limits(mem_mb: int, cpu_seconds: int = 120):
    """Cap memory (RLIMIT_AS) and CPU time (RLIMIT_CPU) on Unix.

    The parent-side wall-clock timeout (runner.py) is the primary guard against
    hangs; RLIMIT_CPU is a belt-and-suspenders cap on busy-loops, and RLIMIT_AS
    stops memory bombs. Output-size is NOT hard-capped here — the parent's
    timeout bounds how long a bot can spam, which is sufficient under the
    trusted-local model; a true output cap is deferred (see spec Trust model).
    """
    try:
        import resource
    except ImportError:
        return  # non-Unix; rely on parent-side timeout only
    limits = [
        (getattr(resource, "RLIMIT_AS", None), mem_mb * 1024 * 1024),
        (getattr(resource, "RLIMIT_CPU", None), cpu_seconds),
    ]
    for res, value in limits:
        if res is not None:
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass


def main(argv):
    try:
        bot_path, seeds_csv = argv[1], argv[2]
        inv_max, mem_mb = float(argv[3]), int(argv[4])
    except (IndexError, ValueError) as e:
        print(BotResult.failed(f"bad arguments: {type(e).__name__}: {e}").to_json())
        return 0
    _apply_limits(mem_mb)

    code = open(bot_path).read()
    ok, detail = validate_code(code)
    if not ok:
        print(BotResult.failed(f"validation: {detail}").to_json())
        return 0

    seeds = [int(s) for s in seeds_csv.split(",") if s != ""]
    result = score_bot(bot_path, seeds, invalid_rate_max=inv_max)
    print(result.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
