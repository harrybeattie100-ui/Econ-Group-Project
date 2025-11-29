"""
Helper utility to bundle the Crisis Forecaster project for FIN41660 submission.

Creates a ZIP archive with code, requirements, and report outputs using only the
standard library. Run from the project root: `python make_submission_zip.py`.
"""

from __future__ import annotations

from pathlib import Path
import zipfile


BASE_DIR = Path(__file__).resolve().parent
TARGET_ZIP = BASE_DIR / "Crisis_Forecaster_Submission.zip"


def _require_dirs(dirs: list[Path]) -> None:
    missing = [d for d in dirs if not d.exists()]
    if missing:
        missing_str = ", ".join(str(d) for d in missing)
        raise FileNotFoundError(f"Required directories missing: {missing_str}")


def collect_files() -> list[Path]:
    """Gather project assets to include in the submission archive."""
    src_dir = BASE_DIR / "src"
    app_file = BASE_DIR / "app" / "app.py"
    report_dir = BASE_DIR / "report"
    tests_dir = BASE_DIR / "tests"
    data_dir = BASE_DIR / "data"

    _require_dirs([src_dir, app_file.parent, report_dir, tests_dir])

    files: set[Path] = set()
    files.update(src_dir.rglob("*.py"))
    files.add(app_file)
    files.add(BASE_DIR / "forecasting_script.py")
    files.add(BASE_DIR / "requirements.txt")
    files.add(BASE_DIR / "README.md")
    files.update(tests_dir.rglob("*.py"))

    if report_dir.exists():
        files.update(report_dir.rglob("*"))

    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            files.add(csv_file)

    return sorted(f for f in files if f.is_file())


def create_zip(files: list[Path]) -> Path:
    """Write the collected files into the submission ZIP."""
    with zipfile.ZipFile(TARGET_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = file_path.relative_to(BASE_DIR)
            zf.write(file_path, arcname)
    return TARGET_ZIP


def main() -> None:
    files = collect_files()
    zip_path = create_zip(files)
    print(f"Created submission archive at {zip_path}")


if __name__ == "__main__":
    main()
