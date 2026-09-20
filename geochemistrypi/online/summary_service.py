"""Bundle existing Online result files without recomputing or transforming them."""

from __future__ import annotations

import hashlib
import re
import shutil
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .artifact_paths import _plain_stat, output_directory, resolve_artifact_path


SUMMARY_ARCHIVE = re.compile(r"summary(?:_[0-9a-f]{64})?\.zip")


def _result_files(output: Path, directory: Path) -> list[str]:
    """List regular outputs, rejecting links before descending into directories."""
    names = []
    for path in sorted(directory.iterdir()):
        info = _plain_stat(path)
        if stat.S_ISDIR(info.st_mode):
            names.extend(_result_files(output, path))
        elif stat.S_ISREG(info.st_mode):
            if path.parent == output and SUMMARY_ARCHIVE.fullmatch(path.name):
                continue  # Never include a previously generated Summary in itself.
            names.append(path.relative_to(output).as_posix())
    return names


def build_summary(jobs_dir: Path, job_id: str) -> dict[str, Any]:
    output = output_directory(jobs_dir, job_id)
    names = _result_files(output, output)
    if not names:
        raise FileNotFoundError("No result files available")

    # Stage outside the downloadable directory and publish only a complete ZIP.
    with TemporaryDirectory(prefix=".summary-build-", dir=output.parent) as staging:
        staged_archive = Path(staging) / "summary.zip"
        with ZipFile(staged_archive, "w") as archive:
            for name in names:
                source = resolve_artifact_path(jobs_dir, job_id, name)
                entry = ZipInfo("summary/" + name)  # Stable timestamps for repeat downloads.
                entry.compress_type = ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                with source.open("rb") as stream, archive.open(entry, "w", force_zip64=True) as target:
                    shutil.copyfileobj(stream, target, length=1024 * 1024)
        with staged_archive.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        # Identical outputs reuse one filename; changed outputs get a new URL.
        archive_path = output / f"summary_{fingerprint}.zip"
        staged_archive.replace(archive_path)

    return {
        "artifact": {
            "name": "summary.zip",
            "download_url": f"/api/data-mining/jobs/{job_id}/files/{archive_path.name}",
            "size_bytes": archive_path.stat().st_size,
        },
        "files": names,
    }
