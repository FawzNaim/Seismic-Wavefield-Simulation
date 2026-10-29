"""Check current snapshot hashes and parse all Python files; no numerical execution."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
failures = []
for record in manifest:
    path = root / record["path"]
    if not path.is_file():
        failures.append(f"Missing: {record['path']}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        failures.append(f"Changed since snapshot: {record['path']}")
python_files = sorted(p for p in root.rglob("*.py")
                      if not any(part in {".git", ".venv", "venv", "__pycache__"}
                                 for part in p.relative_to(root).parts))
for path in python_files:
    try:
        ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as error:
        failures.append(f"Syntax error: {path.relative_to(root)}: {error}")
if failures:
    raise SystemExit("\n".join(failures))
print(f"PASS: {len(manifest)} snapshot hashes; {len(python_files)} Python files parse.")
print("Historical source hashes are retained metadata, not independently verified ownership.")
print("This is a packaging check, not a numerical accuracy test.")
