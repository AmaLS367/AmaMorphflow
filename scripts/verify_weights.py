from __future__ import annotations

import sys
from pathlib import Path


EXPECTED_FILES = [
    "stable-diffusion-v1-5/model_index.json",
    "stable-diffusion-v1-5/v1-inference.yaml",
    "stable-diffusion-v1-5/unet/config.json",
    "stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin",
    "stable-diffusion-v1-5/feature_extractor/preprocessor_config.json",
    "image_encoder/config.json",
    "image_encoder/pytorch_model.bin",
    "sd-vae-ft-mse/config.json",
    "sd-vae-ft-mse/diffusion_pytorch_model.bin",
    "DWPose/dw-ll_ucoco_384.onnx",
    "DWPose/yolox_l.onnx",
    "denoising_unet.pth",
    "motion_module.pth",
    "pose_guider.pth",
    "reference_unet.pth",
    "stable-diffusion-v1-5/tokenizer/merges.txt",
    "stable-diffusion-v1-5/tokenizer/special_tokens_map.json",
    "stable-diffusion-v1-5/tokenizer/tokenizer_config.json",
    "stable-diffusion-v1-5/tokenizer/vocab.json",
    "stable-diffusion-v1-5/scheduler/scheduler_config.json",
    "stable-diffusion-v1-5/text_encoder/config.json",
    "stable-diffusion-v1-5/text_encoder/pytorch_model.bin",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "pretrained_weights"
    missing = []
    empty = []

    for relative_path in EXPECTED_FILES:
        path = root / relative_path
        if not path.exists():
            missing.append(relative_path)
        elif path.stat().st_size <= 0:
            empty.append(relative_path)

    if missing or empty:
        if missing:
            print("Missing files:")
            for relative_path in missing:
                print(f"- {relative_path}")
        if empty:
            print("Empty files:")
            for relative_path in empty:
                print(f"- {relative_path}")
        sys.exit(1)

    print(f"All {len(EXPECTED_FILES)} weight files are present and non-empty.")


if __name__ == "__main__":
    main()
