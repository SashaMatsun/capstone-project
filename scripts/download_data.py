"""Download and verify the Kang PBMC data. Run --help for available modes."""

import argparse
import csv
import hashlib
import http.client
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "metadata/sources.tsv"
LOCK = ROOT / "metadata/checksums.sha256"


def hashes(path):
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def fetch(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "kang-pbmc-capstone/1.0"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            return
        except (OSError, http.client.HTTPException) as exc:
            if attempt == 3:
                raise
            print(f"Transfer failed ({exc}); retrying.", file=sys.stderr)
            time.sleep(2 * attempt)


def make_read_only(path):
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def run(args):
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        source, = csv.DictReader(stream, delimiter="\t")
    name = source["dest"]
    dest = (ROOT / name).resolve()
    dest.relative_to((ROOT / "data/raw").resolve())
    md5 = source["md5"]
    if not re.fullmatch("[0-9a-f]{32}", md5):
        raise ValueError("Invalid provider MD5 in sources.tsv")

    sha256 = None
    if LOCK.exists():
        sha256, locked_name = LOCK.read_text(encoding="utf-8").strip().split(maxsplit=1)
        if locked_name != name or not re.fullmatch("[0-9a-f]{64}", sha256):
            raise ValueError("Invalid SHA-256 or path in checksums.sha256")
    elif not args.init_checksum:
        raise ValueError("Missing checksums.sha256. Restore it from Git or use --init-checksum for initial setup.")

    def matches(pair):
        return pair[0] == md5 and (sha256 is None or pair[1] == sha256)

    observed = hashes(dest) if dest.is_file() else None
    if observed is None or not matches(observed):
        if args.verify_only:
            raise ValueError(f"{name}: " + ("missing" if observed is None else "checksum mismatch"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            staged = Path(tmp) / "download"
            print(f"Downloading {source['url']}", flush=True)
            fetch(source["url"], staged)
            observed = hashes(staged)
            if not matches(observed):
                raise ValueError(f"Downloaded bytes failed verification. MD5={observed[0]}, SHA-256={observed[1]}")
            # Windows requires the old destination to be writable for replacement.
            old_mode = dest.stat().st_mode if os.name == "nt" and dest.exists() else None
            if old_mode is not None:
                dest.chmod(old_mode | stat.S_IWRITE)
            try:
                staged.replace(dest)
            except OSError:
                if old_mode is not None and dest.exists():
                    dest.chmod(old_mode)
                raise

    if sha256 is None:
        # Exclusive creation prevents accidentally overwriting an existing baseline.
        with LOCK.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{observed[1]}  {name}\n")
        print("Created metadata/checksums.sha256. Review it and commit it to Git.")
    if not args.verify_only:
        make_read_only(dest)
    print(f"OK {name}\nMD5     {observed[0]}\nSHA-256 {observed[1]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--init-checksum", action="store_true",
                       help="initialize the project SHA-256 after provider MD5 verification")
    modes.add_argument("--verify-only", action="store_true",
                       help="check local bytes against both recorded hashes without writing or downloading")
    try:
        run(parser.parse_args())
    except (OSError, ValueError, KeyError, http.client.HTTPException) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
