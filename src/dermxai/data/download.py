"""Download HAM10000 from the Harvard Dataverse API and unzip it into data/raw/HAM10000/.

Safety design:
* Without `confirm=True` this module only LISTS files and sizes; it never downloads by surprise.
* Every file is checked against the MD5 checksum that Dataverse publishes.
* Archives are treated as data: only .jpg/.png members are extracted, by base name only,
  so a malicious path inside a zip ("../../x") can never write outside the target folder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from dermxai.utils import GB, check_disk

logger = logging.getLogger(__name__)

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/DBW86T"  # HAM10000 dataset (Tschandl et al., 2018)
# Dataverse's file page recommends downloading via the API URL "in a Wget command or a download
# manager". Its web server rejects the anonymous default client name "python-requests/x.y", so we
# send an honest name that identifies this tool. We do NOT pretend to be a web browser.
USER_AGENT = "dermxai-research-downloader/0.1 (HAM10000 academic project; non-browser script)"
HEADERS = {"User-Agent": USER_AGENT}

# The four files this project needs (the ISIC 2018 test files in the same dataset are not used).
WANTED = {
    "HAM10000_images_part_1.zip": "images",
    "HAM10000_images_part_2.zip": "images",
    "HAM10000_segmentations_lesion_tschandl.zip": "masks",
    "HAM10000_metadata.tab": "metadata",
}
METADATA_CSV = "HAM10000_metadata.csv"


@dataclass(frozen=True)
class RemoteFile:
    """One file in the Dataverse dataset."""

    file_id: int
    name: str  # name as stored on Dataverse
    local_name: str  # name we save it under
    size: int  # bytes
    md5: str
    kind: str  # "images" | "masks" | "metadata"


LISTING_JSON = "dataverse_listing.json"  # saved API response (file ids, sizes, MD5s)


def list_remote_files(timeout: float = 60.0) -> list[RemoteFile]:
    """Ask the Dataverse API which files the HAM10000 dataset has; return the ones we need."""
    url = f"{DATAVERSE}/api/datasets/:persistentId/"
    resp = requests.get(url, params={"persistentId": DOI}, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return _parse_files(resp.json()["data"]["latestVersion"]["files"])


def files_from_saved_listing(raw_dir: Path) -> list[RemoteFile]:
    """Read the needed files (with sizes and MD5s) from the saved API response; no network."""
    listing = json.loads((raw_dir / LISTING_JSON).read_text(encoding="utf-8-sig"))
    return _parse_files(listing["data"]["latestVersion"]["files"])


def _parse_files(files: list[dict]) -> list[RemoteFile]:
    """Pick the four needed files out of a Dataverse 'files' list."""
    out: list[RemoteFile] = []
    for entry in files:
        df = entry["dataFile"]
        name = df["filename"]
        if name not in WANTED or entry.get("restricted", False):
            continue
        kind = WANTED[name]
        # Dataverse converts CSVs into ".tab"; we request the original CSV, whose size is originalFileSize.
        is_tab = kind == "metadata"
        out.append(RemoteFile(
            file_id=int(df["id"]),
            name=name,
            local_name=METADATA_CSV if is_tab else name,
            size=int(df.get("originalFileSize", df["filesize"]) if is_tab else df["filesize"]),
            md5=df["md5"],
            kind=kind,
        ))
    missing = set(WANTED) - {f.name for f in out}
    if missing:
        raise RuntimeError(f"Dataverse listing is missing expected files: {sorted(missing)}")
    return out


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """MD5 hex digest of a file, read in 1 MB chunks (never the whole file in RAM)."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _fetch_into_part(url: str, params: dict, part: Path, rf: RemoteFile, timeout: float) -> None:
    """One attempt: continue the .part file from where it stopped (HTTP Range request)."""
    have = part.stat().st_size if part.exists() else 0
    if have >= rf.size:  # already complete; the MD5 check decides if it is good
        return
    headers = dict(HEADERS)
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, params=params, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        if have and resp.status_code != 206:  # server ignored the Range header: start over
            logger.warning("  server did not resume; restarting %s from 0", rf.name)
            have = 0
        mode = "ab" if have else "wb"
        t0, got, last_log = time.perf_counter(), 0, time.perf_counter()
        with part.open(mode) as fh:
            for block in resp.iter_content(chunk_size=1 << 20):
                fh.write(block)
                got += len(block)
                now = time.perf_counter()
                if now - last_log >= 30 or have + got >= rf.size:
                    speed = got / max(now - t0, 1e-6)
                    eta_min = (rf.size - have - got) / max(speed, 1) / 60
                    logger.info("  %s: %5.1f%% (%.0f / %.0f MB), %.2f MB/s, ETA %.0f min", rf.name,
                                100 * (have + got) / rf.size, (have + got) / 2**20, rf.size / 2**20,
                                speed / 2**20, eta_min)
                    last_log = now


def _download_one(rf: RemoteFile, dest: Path, timeout: float, max_attempts: int = 20) -> None:
    """Download one file to dest via a resumable .part file, retrying stalls, then verify its MD5."""
    if dest.exists() and md5sum(dest) == rf.md5:
        logger.info("Already downloaded and verified: %s", dest.name)
        return
    url = f"{DATAVERSE}/api/access/datafile/{rf.file_id}"
    params = {"format": "original"} if rf.kind == "metadata" else {}
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s (%.1f MB) ...", rf.name, rf.size / 2**20)
    for attempt in range(1, max_attempts + 1):
        try:
            _fetch_into_part(url, params, part, rf, timeout)
            break
        except (requests.ConnectionError, requests.Timeout) as err:
            have = part.stat().st_size / 2**20 if part.exists() else 0
            wait = min(60, 5 * attempt)
            logger.warning("  attempt %d/%d stalled at %.0f MB (%s); resuming in %d s",
                           attempt, max_attempts, have, type(err).__name__, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"Gave up on {rf.name} after {max_attempts} attempts; re-run to resume from {part}")
    got = md5sum(part)
    if got != rf.md5:
        raise RuntimeError(f"Checksum mismatch for {rf.name}: expected {rf.md5}, got {got}. "
                           f"Partial file kept at {part} for inspection.")
    part.replace(dest)
    logger.info("Verified MD5 for %s", dest.name)


def download(raw_dir: Path, min_free_gb: float, confirm: bool, timeout: float = 60.0) -> list[RemoteFile]:
    """List the needed files; if confirm=True, download them into raw_dir/zips (resumable by file)."""
    files = list_remote_files()
    total = sum(f.size for f in files)
    for f in files:
        logger.info("  %-45s %8.1f MB", f.name, f.size / 2**20)
    logger.info("Total download: %.2f GB", total / GB)
    if not confirm:
        logger.info("Listing only. Re-run with --yes to download.")
        return files
    zips = raw_dir / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    # Need space for the zips and roughly the same again once they are extracted.
    free = check_disk(zips, min_free_gb, needed_bytes=2 * total)
    logger.info("Free disk before download: %.1f GB", free)
    for f in files:
        _download_one(f, zips / f.local_name, timeout)
    meta_src = zips / METADATA_CSV
    meta_dst = raw_dir / METADATA_CSV
    if not meta_dst.exists():
        meta_dst.write_bytes(meta_src.read_bytes())
    return files


def verify_local_files(raw_dir: Path) -> None:
    """Check manually downloaded files in raw_dir/zips against the published MD5s; raise on any problem."""
    zips = raw_dir / "zips"
    problems: list[str] = []
    for rf in files_from_saved_listing(raw_dir):
        path = zips / rf.local_name
        if not path.exists():
            problems.append(f"missing: {path}")
            continue
        got = md5sum(path)
        status = "OK" if got == rf.md5 else f"MD5 MISMATCH (expected {rf.md5}, got {got})"
        logger.info("  %-45s %8.1f MB  %s", rf.local_name, path.stat().st_size / 2**20, status)
        if got != rf.md5:
            problems.append(f"checksum mismatch: {path}")
    if problems:
        raise RuntimeError("Local files are not usable:\n  " + "\n  ".join(problems))
    meta_dst = raw_dir / METADATA_CSV
    if not meta_dst.exists():
        meta_dst.write_bytes((zips / METADATA_CSV).read_bytes())
    logger.info("All local files match the Dataverse checksums.")


def extract(raw_dir: Path, min_free_gb: float) -> dict[str, int]:
    """Unzip images into raw_dir/images and masks into raw_dir/masks; return files per folder."""
    zips = raw_dir / "zips"
    counts: dict[str, int] = {}
    for name, kind in WANTED.items():
        if kind == "metadata":
            continue
        target = raw_dir / kind
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips / name) as zf:
            members = [m for m in zf.infolist()
                       if not m.is_dir() and Path(m.filename).suffix.lower() in {".jpg", ".png"}]
            check_disk(target, min_free_gb, needed_bytes=sum(m.file_size for m in members))
            written = 0
            for m in members:
                out = target / Path(m.filename).name  # base name only: zip-slip safe
                if out.exists() and out.stat().st_size == m.file_size:
                    continue
                with zf.open(m) as src, out.open("wb") as dst:
                    dst.write(src.read())
                written += 1
            logger.info("%s: %d members, %d newly extracted into %s", name, len(members), written, target)
        counts[kind] = len(list(target.iterdir()))
    return counts
