"""Check local Markdown links in the project documentation."""

from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def broken_links(root=ROOT):
    documents = [
        root / "README.md",
        root / "PROJECT_HANDOFF.md",
        *(root / "docs").glob("*.md"),
        root / "results/README.md",
    ]
    errors = []
    for path in documents:
        text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for match in re.finditer(r"\]\(([^)]+)\)", text):
            target = match[1]
            url = urlsplit(target)
            if url.scheme or target.startswith(("#", "/")):
                continue
            if not (path.parent / unquote(url.path)).exists():
                errors.append(str(path) + ": " + target)
    return errors


def main():
    errors = broken_links()
    print("\n".join(errors) if errors else "PASS: documentation links")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
