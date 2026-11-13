from pathlib import Path
import re
import sys
from urllib.parse import urlsplit,unquote
root=Path(__file__).resolve().parents[2]
files=[root/'README.md',root/'PROJECT_HANDOFF.md',*(root/'docs').glob('*.md'),root/'results/README.md']
errors=[]
for p in files:
    text=re.sub(r'```.*?```','',p.read_text(),flags=re.S)
    for m in re.finditer(r'\]\(([^)]+)\)',text):
        raw=m[1];u=urlsplit(raw)
        if u.scheme or raw.startswith(('#','/')):continue
        if not (p.parent/unquote(u.path)).exists():errors.append(str(p)+': '+raw)
print('\n'.join(errors) if errors else 'PASS: consolidated documentation links')
sys.exit(bool(errors))
