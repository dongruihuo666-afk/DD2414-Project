"""Print publication commands for reviewed summaries; publish with --execute."""

import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("results/benchmarks.json", "PROJECT_HANDOFF.md")
EXPERIMENT_BRANCHES = (
    "dd2414-mini-baseline",
    "experiment/hybrid-teacher-10sweep-cluster",
    "experiment/rpl-full-data-double-distill",
    "experiment/rpl-depth-10sweep-velocity-study",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", choices=ALLOWED, required=True)
    parser.add_argument("--message", default="Record reviewed experiment summary")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    commands = [
        ["git", "add", "--", *args.path],
        ["git", "commit", "--only", "-m", args.message, "--", *args.path],
    ]
    for command in commands:
        print(shlex.join(command))
    print("git push origin <current-experiment-branch>")
    if not args.execute:
        return

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True,
    ).strip()
    if branch not in EXPERIMENT_BRANCHES:
        raise RuntimeError("Select an experiment branch before publishing")
    for path in args.path:
        value = ROOT / path
        if not value.is_file() or value.is_symlink():
            raise ValueError("Expected a reviewed regular summary file: " + path)

    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", branch], cwd=ROOT, check=True)
    local = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/" + branch],
        cwd=ROOT, text=True,
    ).split()[0]
    if local != remote:
        raise RuntimeError("Remote publication ref mismatch")


if __name__ == "__main__":
    main()
