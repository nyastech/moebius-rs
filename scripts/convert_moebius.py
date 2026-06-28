#!/usr/bin/env python3
"""Convert upstream Moebius/PyTorch artifacts into Candle-friendly files.

This script is intentionally separate from the Rust build.  It requires the
upstream Python stack (`torch`, `diffusers`, `safetensors`, etc.) and writes
generated artifacts under `public/models/moebius-ft-places2` plus parity
fixtures under `tmp/fixtures/moebius-ft-places2`.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any


def require_imports() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from safetensors.torch import save_file
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Missing Python dependency. Run this script through `uv run` with "
            "the conversion dependencies before running "
            f"this script. Original error: {error}"
        ) from error
    if not hasattr(torch, "xpu"):
        # Diffusers 0.38 probes `torch.xpu.empty_cache` during import.  Older
        # CPU-only macOS PyTorch wheels do not expose the namespace even though
        # the Moebius export path never touches XPU devices.
        class _NoXpu:
            @staticmethod
            def empty_cache() -> None:
                return None

            @staticmethod
            def device_count() -> int:
                return 0

            @staticmethod
            def manual_seed(_seed: int) -> None:
                return None

            @staticmethod
            def manual_seed_all(_seed: int) -> None:
                return None

            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def _is_in_bad_fork() -> bool:
                return False

        torch.xpu = _NoXpu()
    if not hasattr(torch.distributed, "device_mesh"):
        # Diffusers 0.38 type annotations refer to this module at import time.
        # PyTorch 2.2.2 is the newest macOS x86_64 wheel available locally and
        # does not expose it, but the CPU conversion path never constructs a
        # distributed mesh.
        class _NoDeviceMesh:
            class DeviceMesh:
                pass

        torch.distributed.device_mesh = _NoDeviceMesh
    return np, torch, save_file


def add_upstream_to_path(upstream: Path) -> None:
    resolved = str(upstream.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_upstream_model(upstream: Path, config: Path, weights: Path, device: str):
    add_upstream_to_path(upstream)
    import yaml

    np, torch, _save_file = require_imports()
    _ = np
    install_minimal_model_lib(upstream)

    with config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model_cfg = dict(cfg["model"])
    model_type = model_cfg.pop("model_type")
    model_cfg["sample_size"] = cfg["data"]["image_size"] // cfg["vae"]["downsample_ratio"]
    model_cfg["num_embeddings"] = 20

    from model_lib.nets import unet_lambda_prune_lite

    diff_cls = getattr(unet_lambda_prune_lite, model_type)
    diff = diff_cls(**model_cfg)
    embedding_dim = diff.config.encoder_hid_dim
    model = RemovalModel(torch, diff, 20, embedding_dim).to(device=device, dtype=torch.float32)
    state_dict = torch.load(str(weights), map_location=device)
    message = model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[load] removal model: {message}")
    return model


def install_minimal_model_lib(upstream: Path) -> None:
    model_lib = types.ModuleType("model_lib")
    model_lib.__path__ = [str(upstream / "model_lib")]
    nets = types.ModuleType("model_lib.nets")
    nets.__path__ = [str(upstream / "model_lib" / "nets")]
    sys.modules.setdefault("model_lib", model_lib)
    sys.modules.setdefault("model_lib.nets", nets)


def RemovalModel(torch, diff_model, num_embeddings: int, embedding_dim: int):
    class _RemovalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
            self.diff_model = diff_model
            self.num_embeddings = num_embeddings

        def forward(self, noisy_latents, timesteps, input_ids):
            encoder_hidden_states = self.embedding_layer(input_ids)
            return self.diff_model(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
            )

    return _RemovalModel()


def load_vae(vae_dir: Path, device: str):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(str(vae_dir)).to(device)
    vae.eval()
    print(f"[load] vae scaling_factor={vae.config.scaling_factor}")
    return vae


def cpu_state_dict(module) -> dict[str, Any]:
    return {key: value.detach().cpu().contiguous() for key, value in module.state_dict().items()}


def write_manifest(path: Path, tensors: dict[str, Any], metadata: dict[str, Any]) -> None:
    manifest = {
        "metadata": metadata,
        "tensors": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
            }
            for key, value in sorted(tensors.items())
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def dump_fixture(np, torch, out_dir: Path, removal_model, vae, device: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "unet_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    image = torch.linspace(-1.0, 1.0, 3 * 512 * 512, device=device, dtype=torch.float32).reshape(
        1, 3, 512, 512
    )
    latent9 = torch.randn(2, 9, 64, 64, device=device, dtype=torch.float32)
    timesteps = torch.tensor([900, 900], device=device, dtype=torch.int64)
    input_ids = torch.tensor(
        [list(range(10, 20)), list(range(0, 10))], device=device, dtype=torch.int64
    )

    trace_handles = install_trace_hooks(np, removal_model, trace_dir)
    with torch.no_grad():
        moments = vae.encode(image).latent_dist.mean * vae.config.scaling_factor
        noise = removal_model(latent9, timesteps=timesteps, input_ids=input_ids).sample
        decoded = vae.decode(moments / vae.config.scaling_factor).sample
    for handle in trace_handles:
        handle.remove()

    np.save(out_dir / "unet_input_latent9.npy", latent9.detach().cpu().numpy())
    np.save(out_dir / "unet_input_timesteps.npy", timesteps.detach().cpu().numpy())
    np.save(out_dir / "unet_input_ids.npy", input_ids.detach().cpu().numpy())
    np.save(out_dir / "vae_encode_mean.npy", moments.detach().cpu().numpy())
    np.save(out_dir / "unet_noise.npy", noise.detach().cpu().numpy())
    np.save(out_dir / "vae_decode.npy", decoded.detach().cpu().numpy())
    print(f"[ok] fixtures: {out_dir}")


def install_trace_hooks(np, removal_model, trace_dir: Path) -> list[Any]:
    diff = removal_model.diff_model
    modules = [
        ("embedding_layer", removal_model.embedding_layer),
        ("diff_model.time_embedding", diff.time_embedding),
        ("diff_model.encoder_hid_proj", getattr(diff, "encoder_hid_proj", None)),
        ("diff_model.conv_in", diff.conv_in),
        ("diff_model.conv_norm_out", diff.conv_norm_out),
        ("diff_model.conv_out", diff.conv_out),
    ]
    modules.extend((f"diff_model.down_blocks.{index}", block) for index, block in enumerate(diff.down_blocks))
    modules.extend((f"diff_model.up_blocks.{index}", block) for index, block in enumerate(diff.up_blocks))
    for block_index, block in enumerate(diff.down_blocks):
        modules.extend(
            (f"diff_model.down_blocks.{block_index}.resnets.{index}", module)
            for index, module in enumerate(getattr(block, "resnets", []))
        )
        modules.extend(
            (f"diff_model.down_blocks.{block_index}.attentions.{index}", module)
            for index, module in enumerate(getattr(block, "attentions", []))
        )
        for attention_index, attention in enumerate(getattr(block, "attentions", [])):
            add_transformer_block_hooks(
                modules,
                f"diff_model.down_blocks.{block_index}.attentions.{attention_index}",
                attention,
            )
        modules.extend(
            (f"diff_model.down_blocks.{block_index}.downsamplers.{index}", module)
            for index, module in enumerate(getattr(block, "downsamplers", []) or [])
        )
    for block_index, block in enumerate(diff.up_blocks):
        modules.extend(
            (f"diff_model.up_blocks.{block_index}.resnets.{index}", module)
            for index, module in enumerate(getattr(block, "resnets", []))
        )
        modules.extend(
            (f"diff_model.up_blocks.{block_index}.attentions.{index}", module)
            for index, module in enumerate(getattr(block, "attentions", []))
        )
        for attention_index, attention in enumerate(getattr(block, "attentions", [])):
            add_transformer_block_hooks(
                modules,
                f"diff_model.up_blocks.{block_index}.attentions.{attention_index}",
                attention,
            )
        modules.extend(
            (f"diff_model.up_blocks.{block_index}.upsamplers.{index}", module)
            for index, module in enumerate(getattr(block, "upsamplers", []) or [])
        )

    def hook_for(name: str):
        safe_name = name.replace(".", "__")

        def _hook(_module, _inputs, output) -> None:
            if ".transformer_blocks." in name and len(_inputs) > 0:
                first_input = _inputs[0]
                np.save(trace_dir / f"{safe_name}__input0.npy", first_input.detach().cpu().numpy())
            tensor = output.sample if hasattr(output, "sample") else output
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            np.save(trace_dir / f"{safe_name}.npy", tensor.detach().cpu().numpy())

        return _hook

    return [module.register_forward_hook(hook_for(name)) for name, module in modules if module is not None]


def add_transformer_block_hooks(modules: list[tuple[str, Any]], prefix: str, attention: Any) -> None:
    for block_index, block in enumerate(getattr(attention, "transformer_blocks", [])):
        block_prefix = f"{prefix}.transformer_blocks.{block_index}"
        modules.extend(
            [
                (f"{block_prefix}.attn1", getattr(block, "attn1", None)),
                (f"{block_prefix}.attn2", getattr(block, "attn2", None)),
                (f"{block_prefix}.ff", getattr(block, "ff", None)),
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("/Users/manish/open_source/Moebius"))
    parser.add_argument("--vae-dir", type=Path, default=Path("tmp/PixelHacker-vae"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("tmp/Moebius-weights/ft_places2/diffusion_pytorch_model.bin"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/Users/manish/open_source/Moebius/config/model_cfg/moebius.yaml"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("public/models/moebius-ft-places2"))
    parser.add_argument("--fixture-dir", type=Path, default=Path("tmp/fixtures/moebius-ft-places2"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np, torch, save_file = require_imports()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    removal_model = load_upstream_model(args.upstream, args.config, args.weights, args.device)
    vae = load_vae(args.vae_dir, args.device)

    moebius_tensors = cpu_state_dict(removal_model)
    vae_tensors = cpu_state_dict(vae)

    save_file(moebius_tensors, args.out_dir / "moebius.safetensors")
    save_file(vae_tensors, args.out_dir / "vae.safetensors")
    write_manifest(
        args.out_dir / "manifest.json",
        {f"moebius::{key}": value for key, value in moebius_tensors.items()}
        | {f"vae::{key}": value for key, value in vae_tensors.items()},
        {
            "model": "Moebius ft_places2",
            "image_size": 512,
            "latent_size": 64,
            "num_embeddings": 20,
            "embedding_dim": 3072,
            "vae_scaling_factor": float(vae.config.scaling_factor),
        },
    )
    dump_fixture(np, torch, args.fixture_dir, removal_model, vae, args.device)
    print(f"[ok] artifacts: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
