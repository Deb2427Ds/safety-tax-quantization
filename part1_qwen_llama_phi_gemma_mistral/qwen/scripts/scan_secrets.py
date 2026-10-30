#!/usr/bin/env python3
"""
scan_secrets.py — pre-run / pre-export secret scanner.
"""
import json
import re
import sys
from pathlib import Path

PATTERNS = [
    ("hf_token_literal",     re.compile(r'hf_[A-Za-z0-9]{20,}')),
    ("google_api_key",       re.compile(r'AIza[A-Za-z0-9_\-]{20,}')),
    ("generic_api_key_kv",   re.compile(r'(?i)(?:api[_-]?key)\s*[=:]\s*["\']?([A-Za-z0-9_\-\.]{12,})["\']?')),
    ("bearer_token",         re.compile(r'(?i)Bearer\s+[A-Za-z0-9_\-\.]{10,}')),
    ("token_assignment",     re.compile(r'(?i)\btoken\s*=\s*["\']([A-Za-z0-9_\-\.]{10,})["\']')),
    ("hf_token_var",         re.compile(r'(?i)HF_TOKEN\s*=\s*["\']([^"\']{4,})["\']')),
    ("gcp_creds_path",       re.compile(r'(?i)GOOGLE_APPLICATION_CREDENTIALS\s*[=:]\s*["\']?([^"\'\s]+)["\']?')),
    ("service_account_path", re.compile(r'["\'][^"\']*service[-_]?account[^"\']*\.json["\']')),
    ("sa_key_path_var",      re.compile(r'(?i)SA_KEY_PATH\s*=\s*["\']([^"\']+)["\']')),
]

ALLOWED_PLACEHOLDERS = {
    "", "none", "null", "...", "your-hf-token-here", "your_hf_token_here",
    "<hf_token>", "<your-token>", "changeme", "redacted", "${hf_token}",
    "os.environ[\"hf_token\"]", "os.environ.get(\"hf_token\")",
}

SAFE_LINE_MARKERS = [
    "os.environ", "os.getenv", "getenv(", "usersecretsclient",
    "get_secret", "load_dotenv", ".env",
]

SCAN_EXTENSIONS = {".py", ".ipynb", ".md", ".json", ".cfg", ".ini", ".yaml", ".yml", ".env.example"}
EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".ipynb_checkpoints", "outputs"}


def iter_text_blobs(path: Path):
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return
        for ci, cell in enumerate(nb.get("cells", [])):
            src = cell.get("source", [])
            if isinstance(src, list):
                src = "".join(src)
            for li, line in enumerate(src.splitlines(), start=1):
                yield f"{path} [cell {ci}]", li, line
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        for li, line in enumerate(text.splitlines(), start=1):
            yield str(path), li, line


def is_placeholder(value: str) -> bool:
    return value.strip().strip("\"'").lower() in ALLOWED_PLACEHOLDERS


def is_safe_line(line: str) -> bool:
    low = line.lower()
    return any(marker in low for marker in SAFE_LINE_MARKERS)


def scan_repo(root: Path):
    findings = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix not in SCAN_EXTENSIONS and path.name != ".env.example":
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        for display_path, line_no, line in iter_text_blobs(path):
            if is_safe_line(line):
                continue
            for name, pattern in PATTERNS:
                m = pattern.search(line)
                if not m:
                    continue
                captured = m.group(1) if m.groups() else m.group(0)
                if is_placeholder(captured):
                    continue
                findings.append((display_path, line_no, name, line.strip()[:160]))
    return findings


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    findings = scan_repo(root)
    if findings:
        print("❌ SECRET SCAN FAILED — possible credentials found:\n")
        for display_path, line_no, name, line in findings:
            print(f"  [{name}] {display_path}:{line_no}\n      {line}\n")
        print(f"Total findings: {len(findings)}")
        sys.exit(1)
    print("✅ scan_secrets.py: no hardcoded credentials found.")
    sys.exit(0)


if __name__ == "__main__":
    main()
