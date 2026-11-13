"""Print a training command from deployment settings; run it with --execute."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    command = [
        config["python"],
        *config["launcher"],
        "-m", "dd2414", config["entry"],
        *config["arguments"],
    ]
    print(shlex.join(command), flush=True)
    if not args.execute:
        return

    root = Path(config["project_root"]).resolve()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True,
    ).strip()
    if branch != config["branch"]:
        raise ValueError("Wrong experiment checkout")
    environment = dict(os.environ, DD2414_PROJECT_ROOT=str(root))
    subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
