"""Modal download entry for InfiniteTalk (Wan2.1 + wav2vec2 + InfiniteTalk weights).

Run:
  modal run download.py::download

Requires Modal secret `huggingface` (HF_TOKEN).
"""

from __future__ import annotations

import os
from typing import Any

import modal

_cfg: dict[str, Any] = {}
volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

WAN_CKPT_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
WAN_CKPT_DIR = f"/models/{WAN_CKPT_REPO}"

WAV2VEC_REPO = "TencentGameMate/chinese-wav2vec2-base"
WAV2VEC_DIR = f"/models/{WAV2VEC_REPO}"
WAV2VEC_SAFETENSORS = "model.safetensors"

INFINITETALK_REPO = "MeiGen-AI/InfiniteTalk"
INFINITETALK_DIR = f"/models/{INFINITETALK_REPO}"
INFINITETALK_SINGLE = "single/infinitetalk.safetensors"

# Comfy workflow: lightx2v distill LoRA + 4 steps (Kijai/WanVideo_comfy).
LIGHTX2V_REPO = "Kijai/WanVideo_comfy"
LIGHTX2V_FILENAME = (
    "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"
)
LIGHTX2V_DIR = f"/models/{LIGHTX2V_REPO}"

model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.34.0,<1.0"),
    volumes={"/models": volume},
    timeout=14400,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def _download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Create Modal secret `huggingface` with your Hugging Face token."
        )

    wan_marker = os.path.join(WAN_CKPT_DIR, "config.json")
    if not os.path.exists(wan_marker):
        print(f"Downloading {WAN_CKPT_REPO} ...")
        snapshot_download(
            repo_id=WAN_CKPT_REPO,
            local_dir=WAN_CKPT_DIR,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Done: {WAN_CKPT_DIR}")
    else:
        print(f"Already exists: {WAN_CKPT_DIR}")

    wav2vec_marker = os.path.join(WAV2VEC_DIR, "config.json")
    if not os.path.exists(wav2vec_marker):
        print(f"Downloading {WAV2VEC_REPO} ...")
        snapshot_download(
            repo_id=WAV2VEC_REPO,
            local_dir=WAV2VEC_DIR,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Done: {WAV2VEC_DIR}")
    else:
        print(f"Already exists: {WAV2VEC_DIR}")

    wav2vec_st = os.path.join(WAV2VEC_DIR, WAV2VEC_SAFETENSORS)
    if not os.path.exists(wav2vec_st) or os.path.getsize(wav2vec_st) < 1000:
        print(f"Downloading {WAV2VEC_REPO}/{WAV2VEC_SAFETENSORS} (pr revision) ...")
        hf_hub_download(
            repo_id=WAV2VEC_REPO,
            filename=WAV2VEC_SAFETENSORS,
            revision="refs/pr/1",
            local_dir=WAV2VEC_DIR,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Done: {wav2vec_st}")

    infinitetalk_weight = os.path.join(INFINITETALK_DIR, INFINITETALK_SINGLE)
    if os.path.exists(infinitetalk_weight) and os.path.getsize(infinitetalk_weight) > 1000:
        print(f"Already exists: {infinitetalk_weight}")
    else:
        print(f"Downloading {INFINITETALK_REPO} ...")
        os.makedirs(INFINITETALK_DIR, exist_ok=True)
        snapshot_download(
            repo_id=INFINITETALK_REPO,
            local_dir=INFINITETALK_DIR,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Done: {infinitetalk_weight}")

    lora_path = os.path.join(LIGHTX2V_DIR, LIGHTX2V_FILENAME)
    if os.path.exists(lora_path) and os.path.getsize(lora_path) > 1000:
        print(f"Already exists: {lora_path}")
    else:
        print(f"Downloading {LIGHTX2V_REPO}/{LIGHTX2V_FILENAME} ...")
        os.makedirs(LIGHTX2V_DIR, exist_ok=True)
        hf_hub_download(
            repo_id=LIGHTX2V_REPO,
            filename=LIGHTX2V_FILENAME,
            local_dir=LIGHTX2V_DIR,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"Done: {lora_path}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
