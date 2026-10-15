"""Verify the pinned GSQ-RCO 3-bit GGUF and write a local MiMo server configuration.

This does not download weights or start an engine. See docs/MIMO.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os

from gguf_reader import GGUFFile
from strata_tokenizer import extract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "build-mimo" / "run")
    parser.add_argument("--exe", type=Path, default=ROOT / "build-mimo" / ("strata-poly.exe" if os.name == "nt" else "strata-poly"))
    parser.add_argument("--context", type=int, choices=(4096, 8192), default=4096)
    parser.add_argument("--skip-hash", action="store_true", help="skip SHA-256 if it was already checked independently")
    args = parser.parse_args()
    target = json.loads((ROOT / "docs" / "mimo-target.json").read_text(encoding="utf-8"))
    spec = target["files"][0]
    model, exe, out = args.gguf.resolve(), args.exe.resolve(), args.out.resolve()
    if not exe.is_file():
        parser.error(f"build the experimental engine first: {exe}")
    if not model.is_file() or model.stat().st_size != spec["size"]:
        parser.error(f"expected the complete {spec['path']} ({spec['size']} bytes)")
    if not args.skip_hash:
        print("Checking SHA-256 of the 115.7 GB file...", flush=True)
        digest = hashlib.sha256()
        with model.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != spec["sha256"]:
            parser.error("SHA-256 does not match the pinned target")
    metadata = GGUFFile(model).metadata
    if metadata.get("general.architecture") != "mimo2" or metadata.get("mimo2.block_count") != 48:
        parser.error("expected the 48-layer MiMo2 text trunk")
    out.mkdir(parents=True, exist_ok=True)
    extract(model, out)
    config = {
        "exe": str(exe), "cwd": str(ROOT),
        "args": ["--model", str(model), "--max-context", str(args.context),
                 "--prompt-batch", "64", "--stage-mib", "512", "--vram-reserve-mib", "1024",
                 "--no-draft", "--window", "1", "--lookup", "0", "--expert-swaps", "0",
                 "--prompt-cache", "0", "--conversation-cache-mib", "0",
                 "--expert-profile", str(out / "mimo.profile")],
        "tokenizer": str(out / "tokenizer"),
        "model_name": "mimo-v2.6-flash-rl-gsq-3bit",
        "log": str(out / "engine.log"),
    }
    config_path = out / "mimo.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {config_path}")
    print(f'python serve/server.py --engine strata --config "{config_path}" --host 127.0.0.1 --port 8096')


if __name__ == "__main__":
    main()
