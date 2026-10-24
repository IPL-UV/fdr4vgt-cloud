"""Lightweight secret scan using regex.

This is intentionally simple: it looks for a short allow-listed set of
high-confidence secret patterns. It is not a replacement for a dedicated
secret-management policy, but it keeps CI green without requiring a paid
license.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Patterns are conservative: high-confidence indicators only.
PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN (RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY-----"),
    "github_pat": re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    "gitlab_pat": re.compile(r"glpat-[a-zA-Z0-9\-]{20}"),
    "openai_key": re.compile(r"sk-[a-zA-Z0-9]{48}"),
    "generic_secret": re.compile(
        r"(api[_-]?key|password|secret|token)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]",
        re.IGNORECASE,
    ),
}

# Paths that are not source code.
SKIP_EXTENSIONS = {".pt2", ".safetensor", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
    "fdr4vgt_model",
}


def should_scan(path: Path) -> bool:
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return False
    for part in path.parts:
        if part in SKIP_DIRS:
            return False
    return True


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    matches: list[tuple[Path, int, str, str]] = []

    for path in root.rglob("*"):
        if not path.is_file() or not should_scan(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS.items():
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    # Allow common placeholder/example values.
                    lowered = line.lower()
                    if any(p in lowered for p in ("example", "placeholder", "your_", "xxx", "***")):
                        continue
                    matches.append((path.relative_to(root), line_no, name, line.strip()))

    if matches:
        print("Potential secrets detected:")
        for path, line_no, name, line in matches:
            print(f"  {path}:{line_no} [{name}] {line}")
        return 1

    print("No obvious secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
