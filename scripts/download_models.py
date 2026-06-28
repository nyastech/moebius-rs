#!/usr/bin/env python3
"""Download project-local Moebius support artifacts.

The Moebius checkpoints are already expected at `tmp/Moebius-weights`.  The
official repo also requires the PixelHacker VAE, so this script downloads only
the missing VAE files into `tmp/PixelHacker-vae`.
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path


VAE_FILES = {
    "config.json": "https://huggingface.co/hustvl/PixelHacker/resolve/main/vae/config.json",
    "diffusion_pytorch_model.bin": "https://huggingface.co/hustvl/PixelHacker/resolve/main/vae/diffusion_pytorch_model.bin",
}


def download(url: str, path: Path, insecure_tls: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"[skip] {path}")
        return

    tmp_path = path.with_suffix(path.suffix + ".part")
    print(f"[get]  {url}")
    context = ssl._create_unverified_context() if insecure_tls else None
    with urllib.request.urlopen(url, context=context) as response, tmp_path.open("wb") as out:
        total_raw = response.headers.get("Content-Length")
        total = int(total_raw) if total_raw else None
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            if total:
                pct = copied / total * 100
                print(f"\r       {copied / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="")
        if total:
            print()
    tmp_path.replace(path)
    print(f"[ok]   {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vae-dir",
        type=Path,
        default=Path("tmp/PixelHacker-vae"),
        help="Directory that will contain PixelHacker VAE config and weights.",
    )
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help="Disable TLS certificate verification when the local Python cert store is broken.",
    )
    args = parser.parse_args()

    for filename, url in VAE_FILES.items():
        download(url, args.vae_dir / filename, args.insecure_tls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
