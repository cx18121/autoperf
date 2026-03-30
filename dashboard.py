"""Dashboard — plot optimization progress from results.tsv."""

import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(path: Path = Path("results.tsv")) -> list[dict]:
    if not path.exists():
        print(f"No results file found at {path}")
        sys.exit(1)
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    results = load_results()
    if not results:
        print("No results to plot.")
        return

    attempts = []
    times = []
    colors = []

    for row in results:
        attempt = int(row["attempt"])
        ms = float(row["ms"])
        status = row["status"].strip()

        attempts.append(attempt)
        times.append(ms)

        if status == "commit" or status == "baseline":
            colors.append("#22c55e")  # green
        elif status == "revert":
            colors.append("#ef4444")  # red
        else:
            colors.append("#94a3b8")  # gray for skips

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot all points
    ax.scatter(attempts, times, c=colors, s=40, zorder=3, edgecolors="white", linewidths=0.5)

    # Connect the committed (best) times with a line
    commit_attempts = []
    commit_times = []
    for a, t, c in zip(attempts, times, colors):
        if c == "#22c55e":
            commit_attempts.append(a)
            commit_times.append(t)

    if commit_attempts:
        ax.step(commit_attempts, commit_times, where="post", color="#22c55e",
                alpha=0.6, linewidth=1.5, label="best (committed)")

    ax.set_xlabel("Attempt")
    ax.set_ylabel("Time (ms)")
    ax.set_title("autoperf — optimization progress")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate speedup
    if len(commit_times) >= 2:
        speedup = commit_times[0] / commit_times[-1]
        ax.annotate(
            f"{speedup:.1f}x speedup",
            xy=(commit_attempts[-1], commit_times[-1]),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="#22c55e",
            arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.5),
        )

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150)
    print("Saved progress.png")
    plt.show()


if __name__ == "__main__":
    main()
