#!/usr/bin/env python3
import subprocess
from pathlib import Path

def main():
    designs_dir = Path("designs")
    out_dir = Path("src/designs")

    if not designs_dir.is_dir():
        raise SystemExit(f"Missing folder: {designs_dir.resolve()}")

    out_dir.mkdir(parents=True, exist_ok=True)

    ui_files = sorted(designs_dir.rglob("*.ui")) 
    if not ui_files:
        print(f"No .ui files found in {designs_dir}")
        return

    for ui_path in ui_files:
        rel = ui_path.relative_to(designs_dir)
        py_path = (out_dir / rel).with_suffix(".py")
        py_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["pyside6-uic", str(ui_path), "-o", str(py_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"Done. Generated {len(ui_files)} file(s) into {out_dir}")

if __name__ == "__main__":
    main()