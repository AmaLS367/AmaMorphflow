from __future__ import annotations

import sys
from pathlib import Path

import requests


DOWNLOADS = [
    (
        "camenduru/AnimateAnyone",
        "raw/main/stable-diffusion-v1-5/model_index.json",
        "stable-diffusion-v1-5/model_index.json",
    ),
    (
        "camenduru/AnimateAnyone",
        "raw/main/stable-diffusion-v1-5/v1-inference.yaml",
        "stable-diffusion-v1-5/v1-inference.yaml",
    ),
    (
        "camenduru/AnimateAnyone",
        "raw/main/stable-diffusion-v1-5/unet/config.json",
        "stable-diffusion-v1-5/unet/config.json",
    ),
    (
        "camenduru/AnimateAnyone",
        "resolve/main/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin",
        "stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin",
    ),
    (
        "camenduru/AnimateAnyone",
        "raw/main/stable-diffusion-v1-5/feature_extractor/preprocessor_config.json",
        "stable-diffusion-v1-5/feature_extractor/preprocessor_config.json",
    ),
    (
        "camenduru/AnimateAnyone",
        "resolve/main/image_encoder/config.json",
        "image_encoder/config.json",
    ),
    (
        "camenduru/AnimateAnyone",
        "resolve/main/image_encoder/pytorch_model.bin",
        "image_encoder/pytorch_model.bin",
    ),
    (
        "camenduru/AnimateAnyone",
        "resolve/main/sd-vae-ft-mse/config.json",
        "sd-vae-ft-mse/config.json",
    ),
    (
        "camenduru/AnimateAnyone",
        "resolve/main/sd-vae-ft-mse/diffusion_pytorch_model.bin",
        "sd-vae-ft-mse/diffusion_pytorch_model.bin",
    ),
    (
        "camenduru/AnimateAnyone",
        "resolve/main/DWPose/dw-ll_ucoco_384.onnx",
        "DWPose/dw-ll_ucoco_384.onnx",
    ),
    (
        "camenduru/AnimateAnyone",
        "resolve/main/DWPose/yolox_l.onnx",
        "DWPose/yolox_l.onnx",
    ),
    (
        "patrolli/AnimateAnyone",
        "resolve/main/denoising_unet.pth",
        "denoising_unet.pth",
    ),
    (
        "patrolli/AnimateAnyone",
        "resolve/main/motion_module.pth",
        "motion_module.pth",
    ),
    (
        "patrolli/AnimateAnyone",
        "resolve/main/pose_guider.pth",
        "pose_guider.pth",
    ),
    (
        "patrolli/AnimateAnyone",
        "resolve/main/reference_unet.pth",
        "reference_unet.pth",
    ),
    (
        "runwayml/stable-diffusion-v1-5",
        "resolve/main/tokenizer/merges.txt",
        "stable-diffusion-v1-5/tokenizer/merges.txt",
    ),
    (
        "runwayml/stable-diffusion-v1-5",
        "resolve/main/tokenizer/special_tokens_map.json",
        "stable-diffusion-v1-5/tokenizer/special_tokens_map.json",
    ),
    (
        "runwayml/stable-diffusion-v1-5",
        "resolve/main/tokenizer/tokenizer_config.json",
        "stable-diffusion-v1-5/tokenizer/tokenizer_config.json",
    ),
    (
        "runwayml/stable-diffusion-v1-5",
        "resolve/main/tokenizer/vocab.json",
        "stable-diffusion-v1-5/tokenizer/vocab.json",
    ),
    (
        "runwayml/stable-diffusion-v1-5",
        "resolve/main/scheduler/scheduler_config.json",
        "stable-diffusion-v1-5/scheduler/scheduler_config.json",
    ),
    (
        "runwayml/stable-diffusion-v1-5",
        "resolve/main/text_encoder/config.json",
        "stable-diffusion-v1-5/text_encoder/config.json",
    ),
    (
        "runwayml/stable-diffusion-v1-5",
        "resolve/main/text_encoder/pytorch_model.bin",
        "stable-diffusion-v1-5/text_encoder/pytorch_model.bin",
    ),
]

CHUNK_SIZE = 1024 * 1024


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size_bytes}B"


def _print_progress(name: str, downloaded: int, total: int | None) -> None:
    if total:
        percent = downloaded / total * 100
        progress = f"{percent:6.2f}% ({_format_size(downloaded)}/{_format_size(total)})"
    else:
        progress = f"{_format_size(downloaded)}"
    print(f"\r{name}: {progress}", end="", flush=True)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0")) or None
        downloaded = 0

        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                _print_progress(destination.name, downloaded, total)

    print()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = repo_root / "pretrained_weights"
    failed: list[tuple[str, str]] = []

    for repo_id, remote_path, local_path in DOWNLOADS:
        destination = target_root / local_path
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[skip] {local_path}")
            continue
        if destination.exists():
            print(f"[retry] {local_path} exists but is empty")

        url = f"https://huggingface.co/{repo_id}/{remote_path}"
        print(f"[download] {local_path}")
        try:
            _download_file(url, destination)
        except Exception as exc:
            failed.append((local_path, str(exc)))
            if destination.exists():
                destination.unlink()
            print(f"[failed] {local_path}: {exc}")

    if failed:
        print("\nFailed downloads:")
        for local_path, error in failed:
            print(f"- {local_path}: {error}")
        sys.exit(1)

    print("\nAll files are present.")


if __name__ == "__main__":
    main()
