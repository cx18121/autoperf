"""Dashboard — plot 2048 bot optimization progress from results.tsv."""

import sys, csv
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    path = Path("results.tsv")
    if not path.exists():
        sys.exit(f"No results file at {path}")

    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if not rows:
        sys.exit("No results to plot.")

    attempts, scores, colors = [], [], []
    for row in rows:
        attempts.append(int(row["attempt"]))
        scores.append(float(row["score"]))
        status = row["status"].strip()
        if status in ("commit", "baseline"):
            colors.append("#22c55e")
        elif status == "revert":
            colors.append("#ef4444")
        else:
            colors.append("#94a3b8")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(attempts, scores, c=colors, s=40, zorder=3, edgecolors="white", linewidths=0.5)

    # Step line through commits
    ca = [a for a, c in zip(attempts, colors) if c == "#22c55e"]
    cs = [s for s, c in zip(scores, colors) if c == "#22c55e"]
    if ca:
        ax.step(ca, cs, where="post", color="#22c55e", alpha=0.6, linewidth=1.5, label="best (committed)")

    ax.set_xlabel("Attempt")
    ax.set_ylabel("Average Score")
    ax.set_title("autoperf — 2048 bot optimization progress")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if len(cs) >= 2:
        improvement = cs[-1] / cs[0]
        ax.annotate(f"{improvement:.1f}x improvement", xy=(ca[-1], cs[-1]),
                    xytext=(15, 15), textcoords="offset points", fontsize=11,
                    fontweight="bold", color="#22c55e",
                    arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.5))

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150)
    print("Saved progress.png")
    plt.show()


if __name__ == "__main__":
    main()
