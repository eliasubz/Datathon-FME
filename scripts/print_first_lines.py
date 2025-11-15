# %%
#!/usr/bin/env python3
"""Print the first line of data/test.csv and data/train.csv.

Usage: python3 scripts/print_first_lines.py
"""
import sys
from pathlib import Path

# %%
def print_first_line(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"{path}: FILE NOT FOUND", file=sys.stderr)
        return
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            line = f.readline()
            if line == "":
                print(f"{path}: (empty file)")
            else:
                print(f"{path}: {line.rstrip('\n').split(';')}")
    except Exception as e:
        print(f"{path}: ERROR reading file: {e}", file=sys.stderr)


def main() -> int:
    print_first_line("data/test.csv")
    print()
    print_first_line("data/train.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
