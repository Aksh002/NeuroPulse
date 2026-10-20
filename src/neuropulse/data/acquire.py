"""Acquire the NASA-linked, checksum-pinned C-MAPSS benchmark without unsafe extraction."""

import argparse
import hashlib
import io
import json
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
ARCHIVE_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
NESTED_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
NESTED_PATH = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
MEMBERS = {"readme.txt", "Damage Propagation Modeling.pdf"} | {
    f"{kind}_FD00{subset}.txt" for kind in ("train", "test", "RUL") for subset in range(1, 5)
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def inspect_archive(payload: bytes) -> dict[str, bytes]:
    """Verify both archives before reading only the fixed allowlist of regular files."""
    if len(payload) > 50_000_000 or sha256(payload) != ARCHIVE_SHA256:
        raise ValueError("Source archive size/checksum differs from the audited benchmark")
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        info = outer.getinfo(NESTED_PATH)
        if info.file_size > 50_000_000:
            raise ValueError("Nested archive exceeds size limit")
        nested = outer.read(info)
    if sha256(nested) != NESTED_SHA256:
        raise ValueError("Nested archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        infos = archive.infolist()
        if len(infos) != len(MEMBERS) or {i.filename for i in infos} != MEMBERS:
            raise ValueError("Archive member list differs; no paths will be extracted")
        if sum(i.file_size for i in infos) > 200_000_000:
            raise ValueError("Expanded archive exceeds size limit")
        return {i.filename: archive.read(i) for i in infos}


def acquire(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "source.zip"
    if archive_path.exists():
        payload = archive_path.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read(50_000_001)
    files = inspect_archive(payload)
    for name, content in files.items():
        target = output / name
        if target.exists() and sha256(target.read_bytes()) != sha256(content):
            raise ValueError(f"Existing raw file has changed: {name}; refusing overwrite")
    atomic_write(archive_path, payload)
    for name, content in files.items():
        atomic_write(output / name, content)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "dataset": "cmapss",
        "source_url": URL,
        "acquired_at": datetime.now(UTC).isoformat(),
        "archive_sha256": ARCHIVE_SHA256,
        "nested_sha256": NESTED_SHA256,
        "license": "not specified; redistribution unresolved",
        "attribution": "Saxena and Goebel (2008), NASA Prognostics Data Repository",
        "files": {
            name: {"sha256": sha256(content), "bytes": len(content)}
            for name, content in sorted(files.items())
        },
    }
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2).encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/cmapss"))
    args = parser.parse_args()
    print(json.dumps(acquire(args.output), indent=2))


if __name__ == "__main__":
    main()
