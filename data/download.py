"""Download the raw LG 18650HG2 dataset (Kollmeyer et al., McMaster) from Mendeley Data.

Dataset: P. Kollmeyer, M. Naguib, M. Skells, "LG 18650HG2 Li-ion Battery Data",
Mendeley Data, V2, 2020, doi:10.17632/b5mj79w5w9.2. Licence CC BY 4.0.

Only the .mat files are fetched. Per the dataset readme, each .csv holds the same
data with an extra header, and the .mat files are several times smaller.
Raw data is never committed to the repository; it lands in data/raw/ (git-ignored).

Usage:
    python data/download.py                  # all six temperatures
    python data/download.py --temps 25degC   # one folder
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DATASET_ID = "b5mj79w5w9"
VERSION = 2
API = "https://data.mendeley.com/public-api/datasets"
FILES_URL = "https://data.mendeley.com/public-files/datasets/{ds}/files/{fid}/file_downloaded"
TEMP_FOLDERS = ["n20degC", "n10degC", "0degC", "10degC", "25degC", "40degC"]
ROOT_FILES = {
    "Readme file - Description of Experimental Tests.txt",
    "Technical Information and Experimental Test Results for LG 18650HG2.pdf",
}
RAW_DIR = Path(__file__).resolve().parent / "raw" / "LG_HG2"


# Mendeley's file endpoint rejects Python's default "Python-urllib" User-Agent,
# so every request sends an explicit one.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; socest-downloader/0.1)", "Accept": "*/*"}


def _get_json(url: str):
    req = urllib.request.Request(url, headers={**HEADERS, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _name(entry: dict) -> str:
    return entry.get("filename") or entry.get("name")


def _size(entry: dict):
    if "size" in entry:
        return int(entry["size"])
    details = entry.get("content_details") or {}
    return int(details["size"]) if "size" in details else None


def list_files(folder_id: str):
    return _get_json(f"{API}/{DATASET_ID}/files?folder_id={folder_id}&version={VERSION}")


def _candidate_urls(entry: dict):
    """The documented public-files endpoint first, then any direct URL the API listing provides."""
    urls = [FILES_URL.format(ds=DATASET_ID, fid=entry["id"])]
    details = entry.get("content_details") or {}
    for key in ("download_url", "view_url"):
        if details.get(key) and details[key] not in urls:
            urls.append(details[key])
    return urls


def download(entry: dict, dest: Path, retries: int = 3):
    expected = _size(entry)
    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for url in _candidate_urls(entry):
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                if expected is not None and tmp.stat().st_size != expected:
                    raise IOError(f"size mismatch: got {tmp.stat().st_size}, expected {expected}")
                tmp.replace(dest)
                return "downloaded"
            except urllib.error.HTTPError as exc:
                errors.append(f"{url} -> HTTP {exc.code}")
                if exc.code in (401, 403, 404):
                    break  # not transient; try the next URL
            except Exception as exc:  # network errors: retry
                errors.append(f"{url} -> {exc}")
                print(f"  retry {attempt + 1} for {dest.name}: {exc}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {dest.name}:\n  " + "\n  ".join(errors))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temps", nargs="*", default=TEMP_FOLDERS, choices=TEMP_FOLDERS)
    ap.add_argument("--out", type=Path, default=RAW_DIR)
    args = ap.parse_args(argv)

    folders = {f["name"]: f["id"] for f in _get_json(f"{API}/{DATASET_ID}/folders/{VERSION}")}
    missing = [t for t in args.temps if t not in folders]
    if missing:
        sys.exit(f"Folders not found in dataset: {missing}. Available: {sorted(folders)}")

    # The readme and PDF are documentation only; a failure there must not stop the data download.
    for entry in list_files("root"):
        if _name(entry) in ROOT_FILES:
            try:
                print(f"root/{_name(entry)}: {download(entry, args.out / _name(entry))}")
            except RuntimeError as exc:
                print(f"root/{_name(entry)}: skipped ({exc})", file=sys.stderr)

    failed = []
    for temp in args.temps:
        entries = [e for e in list_files(folders[temp]) if _name(e).lower().endswith(".mat")]
        print(f"{temp}: {len(entries)} .mat files")
        for e in entries:
            try:
                print(f"  {_name(e)}: {download(e, args.out / temp / _name(e))}")
            except RuntimeError as exc:
                failed.append(f"{temp}/{_name(e)}")
                print(f"  {_name(e)}: FAILED ({exc})", file=sys.stderr)
                if len(failed) == 3 and not any(p.exists() for p in args.out.rglob("*.mat")):
                    sys.exit("First three data files failed and none succeeded; stopping. See errors above.")
    if failed:
        sys.exit(f"{len(failed)} files failed; rerun to retry (finished files are cached).")
    print("All files downloaded.")


if __name__ == "__main__":
    main()
