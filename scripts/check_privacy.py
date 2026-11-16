"""Privacy audit script: ensures zero proprietary brands, suppliers, or raw descriptions leak into git."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from src.config import DATA_DEMO_DIR, ROOT_DIR


FORBIDDEN_PATTERNS = [
    r"COCOYO",
    r"CAT\s*BEST",
    r"PROHOLO",
    r"SCHESIR",
    r"BIOK",
    r"HITA",
    r"CN133",
    r"DM023",
    r"JP057",
    r"CA085",
    r"IT053",
    r"IT054",
    r"COC400",
    r"JR21742",
    r"LB5540",
    r"SCH750",
    r"貓砂",
    r"衛生尿墊",
    r"成貓糧",
    r"罐頭",
    r"Robyn",
    r"rybon",
    r"Budget Allocator",
]


def audit_directory() -> int:
    """Scan demo data, src, scripts, tests, and README for forbidden strings."""
    targets = []
    for sub in ["src", "scripts", "tests", "data/demo", "data/artifacts"]:
        p = ROOT_DIR / sub
        if p.exists():
            targets.extend([f for f in p.rglob("*") if f.is_file() and f.suffix in {".py", ".csv", ".json", ".md"}])
    for top in ["app.py", "README.md", "requirements.txt", "项目架构.md"]:
        fp = ROOT_DIR / top
        if fp.exists():
            targets.append(fp)

    pattern = re.compile("|".join(FORBIDDEN_PATTERNS), re.IGNORECASE)
    violations = []
    for fp in targets:
        if fp.name == "check_privacy.py":
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore")
        hits = pattern.findall(text)
        if hits:
            violations.append((str(fp.relative_to(ROOT_DIR)), sorted(set(hits))))

    if violations:
        print("PRIVACY / CLEANUP AUDIT FAILED:")
        for rel, hits in violations:
            print(f"  {rel}: {hits}")
        return 1

    print(f"Privacy & clean-architecture audit PASSED across {len(targets)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(audit_directory())
