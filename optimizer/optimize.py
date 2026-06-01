"""Beam/top-K LLM optimization loop for 2048 bots.

generate() is injected so tests can stub the LLM. The saved best_so_far is
tracked separately from the beam and only updated via the screen->holdout
promotion gate, guaranteeing the written best never regresses.
"""

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from optimizer import llm
from optimizer.prompts import mutation_prompt, wildcard_prompt, extract_code
from optimizer.runner import evaluate
from optimizer.harness import BotResult

# --- Config defaults (all overridable via run_optimization args / CLI) ---
ROUNDS = 20
BEAM_K = 3
CHILDREN_PER_PARENT = 2
SCREEN_SEEDS = list(range(30))
HOLDOUT_SEEDS = list(range(1000, 1100))
TEMPERATURE = 0.8
WILDCARD_EVERY = 3
TIMEOUT = 60
MEM_LIMIT_MB = 1024
INVALID_RATE_MAX = 0.05


def normalized_hash(code: str) -> str:
    """SHA-256 of the code's normalized AST dump (ignores comments/whitespace)."""
    try:
        return hashlib.sha256(ast.dump(ast.parse(code)).encode()).hexdigest()
    except SyntaxError:
        return hashlib.sha256(code.encode()).hexdigest()


@dataclass
class Candidate:
    cid: str
    code: str
    strategy: str                 # "seed" | "mutation" | "wildcard"
    parent_id: str | None = None
    screen: BotResult | None = None
    holdout: BotResult | None = None


def _seed_code() -> str:
    return (Path(__file__).parent / "seed_bot.py").read_text()


def run_optimization(
    rounds=ROUNDS, beam_k=BEAM_K, children_per_parent=CHILDREN_PER_PARENT,
    screen_seeds=None, holdout_seeds=None, temperature=TEMPERATURE,
    wildcard_every=WILDCARD_EVERY, timeout=TIMEOUT, mem_limit_mb=MEM_LIMIT_MB,
    invalid_rate_max=INVALID_RATE_MAX, generate=None, output_dir="output",
    seed_code=None, run_id="run",
) -> Candidate:
    screen_seeds = SCREEN_SEEDS if screen_seeds is None else screen_seeds
    holdout_seeds = HOLDOUT_SEEDS if holdout_seeds is None else holdout_seeds
    generate = generate or llm.generate

    out = Path(output_dir)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    log_path = out / "results.jsonl"
    counter = {"n": 0}
    seen_hashes = set()

    def new_id():
        counter["n"] += 1
        return f"c{counter['n']:04d}"

    def write_code(cand: Candidate) -> str:
        p = out / "candidates" / f"{cand.cid}.py"
        p.write_text(cand.code)
        return str(p)

    def score(cand: Candidate, seeds):
        path = write_code(cand)
        return evaluate(path, seeds, timeout=timeout, mem_limit_mb=mem_limit_mb,
                        invalid_rate_max=invalid_rate_max)

    def try_generate(prompt, temp):
        """Call the LLM, but treat any failure as a skipped candidate rather
        than a crash — a mid-run Ollama error must not kill the whole loop."""
        try:
            return generate(prompt, temperature=temp)
        except Exception as e:
            print(f"  generate failed ({type(e).__name__}: {e}); skipping candidate")
            return None

    config = {"beam_k": beam_k, "children_per_parent": children_per_parent,
              "temperature": temperature, "model": llm.MODEL,
              "screen_seeds": list(screen_seeds), "holdout_seeds": list(holdout_seeds)}

    def log(cand: Candidate, round_no: int, became_best: bool):
        rec = {
            "run_id": run_id, "round": round_no, "candidate_id": cand.cid,
            "parent_id": cand.parent_id, "strategy": cand.strategy,
            "code_hash": normalized_hash(cand.code),
            "code_path": f"{output_dir}/candidates/{cand.cid}.py",
            "status": cand.screen.status if cand.screen else "failed",
            "detail": cand.screen.detail if cand.screen else "no score",
            "screen_score": cand.screen.mean_score if cand.screen else 0.0,
            "screen_stderr": cand.screen.score_stderr if cand.screen else 0.0,
            "max_tile": cand.screen.max_tile if cand.screen else 0,
            "invalid_rate": cand.screen.invalid_rate if cand.screen else 0.0,
            "crashes": cand.screen.crashes if cand.screen else 0,
            "holdout_score": cand.holdout.mean_score if cand.holdout else None,
            "became_best": became_best, "config": config,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # --- Baseline: seed bot ---
    def say(msg):
        # Flush so progress is visible live even when stdout is piped/redirected.
        print(msg, flush=True)

    say(f"[seed] scoring baseline on {len(screen_seeds)} screen "
        f"+ {len(holdout_seeds)} holdout games...")
    seed = Candidate(cid=new_id(), code=seed_code or _seed_code(), strategy="seed")
    seed.screen = score(seed, screen_seeds)
    seed.holdout = score(seed, holdout_seeds)
    seen_hashes.add(normalized_hash(seed.code))
    best = seed
    beam = [seed]
    log(seed, 0, became_best=True)
    say(f"[seed] screen={seed.screen.mean_score:.0f} "
        f"holdout={seed.holdout.mean_score:.0f}  <- baseline to beat")

    def history_text(beam):
        lines = []
        for c in beam:
            sc = c.screen.mean_score if c.screen and c.screen.status == "ok" else "failed"
            lines.append(f"- {c.strategy} ({c.cid}): {sc}")
        return "\n".join(lines) or "(none)"

    for rnd in range(1, rounds + 1):
        say(f"\n=== round {rnd}/{rounds} === (best so far: "
            f"{best.strategy} holdout={best.holdout.mean_score:.0f})")
        candidates = []
        # Mutations of each beam member.
        for parent in beam:
            for _ in range(children_per_parent):
                say(f"  generating mutation of {parent.cid}...")
                resp = try_generate(mutation_prompt(parent.code, history_text(beam)),
                                    temperature)
                code = extract_code(resp) if resp else None
                if not code:
                    continue
                h = normalized_hash(code)
                if h in seen_hashes:
                    say("    (duplicate of an earlier candidate, skipped)")
                    continue
                seen_hashes.add(h)
                candidates.append(Candidate(new_id(), code, "mutation", parent.cid))
        # Periodic wildcard.
        if wildcard_every and rnd % wildcard_every == 0:
            say("  generating wildcard (fresh strategy)...")
            resp = try_generate(wildcard_prompt(history_text(beam)),
                                min(1.0, temperature + 0.1))
            code = extract_code(resp) if resp else None
            if code:
                h = normalized_hash(code)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    candidates.append(Candidate(new_id(), code, "wildcard", None))

        # Screen-score new candidates.
        for cand in candidates:
            say(f"  scoring {cand.cid} ({cand.strategy}) on {len(screen_seeds)} games...")
            cand.screen = score(cand, screen_seeds)
            if cand.screen.status == "ok":
                say(f"    {cand.cid}: screen={cand.screen.mean_score:.0f}")
            else:
                say(f"    {cand.cid}: FAILED ({cand.screen.detail[:50]})")

        # Re-rank: keep top-K of (beam + ok candidates) by screen score.
        pool = beam + [c for c in candidates if c.screen and c.screen.status == "ok"]
        pool.sort(key=lambda c: (c.screen.mean_score, -c.screen.invalid_rate),
                  reverse=True)
        beam = pool[:beam_k]

        # Promotion gate: top candidate must beat best on screen AND holdout.
        # pool = beam + ok-candidates, and beam always contains at least the
        # seed, so pool is never empty and every member has been screen-scored.
        assert pool, "pool invariant: beam always retains at least one member"
        top = pool[0]
        assert top.screen is not None and best.screen is not None
        became_best = False
        if top is not best and top.screen.mean_score > best.screen.mean_score:
            say(f"  {top.cid} leads on screen ({top.screen.mean_score:.0f} > "
                f"{best.screen.mean_score:.0f}); confirming on {len(holdout_seeds)} "
                f"holdout games...")
            top.holdout = score(top, holdout_seeds)
            if (top.holdout.status == "ok"
                    and top.holdout.mean_score > best.holdout.mean_score):
                best = top
                became_best = True
                (out / "best_bot.py").write_text(best.code)
                say(f"  *** NEW BEST: {top.cid} holdout={top.holdout.mean_score:.0f} "
                    f"(was {seed.holdout.mean_score:.0f} at seed) -> output/best_bot.py ***")
            else:
                say(f"  {top.cid} did NOT beat best on holdout "
                    f"({top.holdout.mean_score:.0f}); best unchanged")

        # Persist every candidate this round.
        for cand in candidates:
            log(cand, rnd, became_best=(cand is best and became_best))
        # If the promoted best was a retained beam member (not generated this
        # round), it isn't in `candidates` — log the promotion event explicitly
        # so results.jsonl always records when the best changed.
        if became_best and best not in candidates:
            log(best, rnd, became_best=True)

        say(f"round {rnd} done: best={best.strategy}({best.cid}) "
            f"holdout={best.holdout.mean_score:.1f}")

    # Ensure best_bot.py exists even if nothing beat the seed.
    if not (out / "best_bot.py").exists():
        (out / "best_bot.py").write_text(best.code)
    return best


def _benchmark(bot_path: str):
    # No preflight: benchmarking scores a local file and never calls the LLM.
    result = evaluate(bot_path, HOLDOUT_SEEDS, timeout=TIMEOUT,
                      mem_limit_mb=MEM_LIMIT_MB, invalid_rate_max=1.0)
    print(f"{bot_path}: status={result.status} mean={result.mean_score:.1f} "
          f"max_tile={result.max_tile} invalid_rate={result.invalid_rate:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="2048 bot optimizer")
    ap.add_argument("--benchmark", metavar="BOT_PATH",
                    help="score one bot over the holdout set and exit")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    args = ap.parse_args(argv)

    if args.benchmark:
        _benchmark(args.benchmark)
        return
    llm.preflight()  # fail early with a clear message if Ollama isn't ready
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_optimization(rounds=args.rounds, run_id=run_id)


if __name__ == "__main__":
    main()
