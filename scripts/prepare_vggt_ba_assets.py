"""Fetch only audited official BA assets; inference itself remains offline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def fetch(asset, root, record_first_use=False):
    expected = asset["sha256"]
    if not expected and not record_first_use:
        raise ValueError("missing asset pin; first-use recording requires an explicit flag")
    path = root/asset["filename"]
    if Path(asset["filename"]).name != asset["filename"]:
        raise ValueError("asset filename must be a basename")
    started = time.perf_counter()
    digest, count = hashlib.sha256(), 0
    if path.exists():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
                count += len(block)
        if expected and digest.hexdigest() != expected:
            raise ValueError(f"existing asset hash mismatch: {path.name}")
        source = "verified_existing_file"
    else:
        temporary = path.with_suffix(path.suffix+".partial")
        try:
            with urllib.request.urlopen(asset["download_url"], timeout=60) as response, temporary.open("wb") as stream:
                for block in iter(lambda: response.read(1024*1024), b""):
                    count += len(block)
                    if count > 1024**3:
                        raise ValueError("individual asset exceeds 1 GiB transfer cap")
                    digest.update(block)
                    stream.write(block)
            if expected and digest.hexdigest() != expected:
                raise ValueError(f"downloaded asset hash mismatch: {path.name}")
            temporary.replace(path)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        source = "official_download"
    return {**asset, "sha256": digest.hexdigest(), "bytes": count, "preparation_seconds": time.perf_counter()-started,
            "preparation_source": source, "pin_provenance": "previous/publication pin" if expected else "first observed official HTTPS bytes; not publisher-supplied checksum"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record-first-use", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, asset, args.output, args.record_first_use) for asset in config["assets"]]
        assets = [future.result() for future in futures]
    result = {**config, "assets": assets, "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "total_seconds": time.perf_counter()-started}
    (args.output/"assets.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"assets": [{"name": a["name"], "bytes": a["bytes"], "sha256": a["sha256"]} for a in assets], "total_seconds": result["total_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
