"""Modal deploy entry for InfiniteTalk audio-driven lip sync.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import modal
from tongflow import deploy
from tongflow.models.audio_image_gen_video import (
    AudioImageGenVideoInput,
    AudioImageGenVideoOutput,
)
from tongflow.models.audio_video_lip_sync import (
    AudioVideoLipSyncInput,
    AudioVideoLipSyncOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

_cfg: dict[str, Any] = {}
_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

INFINITETALK_ROOT = "/opt/InfiniteTalk"
WAN_CKPT_DIR = "/models/Wan-AI/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR = "/models/TencentGameMate/chinese-wav2vec2-base"
INFINITETALK_WEIGHT = "/models/MeiGen-AI/InfiniteTalk/single/infinitetalk.safetensors"

DEFAULT_PROMPT = (
    "A person speaking naturally to the camera with accurate lip sync and "
    "expressive facial movement."
)
# Align with Comfy workflow JSON + InfiniteTalk README § lightx2v (4-step distill).
DEFAULT_MOTION_FRAME = 9
DEFAULT_SAMPLE_STEPS = 4
DEFAULT_SHIFT = 2.0
DEFAULT_TEXT_GUIDE_SCALE = 1.0
DEFAULT_AUDIO_GUIDE_SCALE = 2.0
DEFAULT_SIZE = "infinitetalk-480"
DEFAULT_FPS = 25.0
DEFAULT_LORA_SCALE = 1.0
# InfiniteTalk generates in 81-frame windows (4n+1). For audio longer than one
# window we run streaming mode, chaining windows so the video tracks the full
# speech length instead of a single ~3.2s clip. A generous safety cap keeps
# runaway audio from exhausting GPU time/memory (override via env).
WINDOW_FRAME_NUM = 81
MAX_VIDEO_S = float(os.environ.get("INFINITETALK_MAX_VIDEO_S", "60"))

LIGHTX2V_LORA_PATH = (
    "/models/Kijai/WanVideo_comfy/"
    "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"
)

# Pin repo for reproducible image builds.
INFINITETALK_GIT_REF = "main"

# Based on InfiniteTalk README (cu121 + flash_attn 2.7.4). Torch 2.5.x is required so
# diffusers 0.38 (via xfuser) can import attention_dispatch custom ops on Modal.
_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
_TORCH_VERSION = "2.5.1"
_TORCHVISION_VERSION = "0.20.1"
_TORCHAUDIO_VERSION = "2.5.1"
_XFORMERS_VERSION = "0.0.29.post1"


def _maybe_bytes(val: object) -> Optional[bytes]:
    if val is None:
        return None
    try:
        return prompt_media_to_bytes(val)
    except (TypeError, ValueError):
        return None


def _probe_duration_seconds(path: str) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
    )
    return max(0.5, float(out.decode().strip()))


def _align_4np1(frames: int) -> int:
    """InfiniteTalk requires frame counts of the form 4n+1; round down (min 17)."""
    n = max(17, int(frames))
    k = (n - 1) // 4
    return max(17, 4 * k + 1)


def _plan_frames(
    duration_s: float, fps: float = DEFAULT_FPS
) -> tuple[int, int, str]:
    """Map audio length to (window_frame_num, max_frames_num, mode).

    Short audio fits one clip; longer audio streams across 81-frame windows so
    the generated video matches the full speech length (capped at MAX_VIDEO_S).
    """
    capped_s = min(duration_s, MAX_VIDEO_S)
    total = _align_4np1(int(capped_s * fps))
    window = min(total, WINDOW_FRAME_NUM)
    mode = "streaming" if total > window else "clip"
    return window, total, mode


def _normalize_audio_to_wav(src: Path, dst: Path) -> None:
    """16 kHz mono PCM — matches InfiniteTalk / wav2vec expectations."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _infinitetalk_cli_args(
    gen: Any,
    *,
    size: str,
    seed: int,
    frame_num: int,
    max_frame_num: int,
    mode: str,
    input_json: str,
    audio_save_dir: str,
) -> Any:
    """Build the same argparse namespace as generate_infinitetalk.py (extra_args=args)."""
    import sys

    argv = [
        "generate_infinitetalk.py",
        "--task",
        "infinitetalk-14B",
        "--ckpt_dir",
        WAN_CKPT_DIR,
        "--wav2vec_dir",
        WAV2VEC_DIR,
        "--infinitetalk_dir",
        INFINITETALK_WEIGHT,
        "--input_json",
        input_json,
        "--size",
        size,
        "--sample_shift",
        str(DEFAULT_SHIFT),
        "--mode",
        mode,
        "--motion_frame",
        str(DEFAULT_MOTION_FRAME),
        "--frame_num",
        str(frame_num),
        "--max_frame_num",
        str(max_frame_num),
        "--sample_steps",
        str(DEFAULT_SAMPLE_STEPS),
        "--sample_text_guide_scale",
        str(DEFAULT_TEXT_GUIDE_SCALE),
        "--sample_audio_guide_scale",
        str(DEFAULT_AUDIO_GUIDE_SCALE),
        "--audio_save_dir",
        audio_save_dir,
        "--base_seed",
        str(seed),
        "--offload_model",
        "False",
        "--lora_dir",
        LIGHTX2V_LORA_PATH,
        "--lora_scale",
        str(DEFAULT_LORA_SCALE),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        return gen._parse_args()
    finally:
        sys.argv = old_argv


def _load_generate_module():
    import importlib.util
    import sys

    name = "generate_infinitetalk"
    if name in sys.modules:
        return sys.modules[name]
    path = f"{INFINITETALK_ROOT}/generate_infinitetalk.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("tongflow==0.1.0")
    .run_commands(
        f"git clone --depth 1 --branch {INFINITETALK_GIT_REF} "
        f"https://github.com/MeiGen-AI/InfiniteTalk.git {INFINITETALK_ROOT}",
    )
    # README § Installation steps 1–3 (cu121 + xformers, then flash-attn)
    .run_commands(
        f"pip install torch=={_TORCH_VERSION} torchvision=={_TORCHVISION_VERSION} "
        f"torchaudio=={_TORCHAUDIO_VERSION} --index-url {_TORCH_INDEX}",
        f"pip install -U xformers=={_XFORMERS_VERSION} --index-url {_TORCH_INDEX}",
    )
    .run_commands(
        "pip install ninja psutil packaging wheel",
        "pip install misaki[en]",
        "pip install flash_attn==2.7.4.post1 --no-build-isolation",
        "pip uninstall -y flash-attn-3 2>/dev/null || true",
        gpu="A10G",
    )
    .run_commands(
        f"cd {INFINITETALK_ROOT} && pip install -r requirements.txt",
        # requirements.txt pulls latest HF libs; keep xfuser-compatible diffusers, cap transformers.
        "pip install 'transformers==4.49.0' 'tokenizers==0.21.0'",
        "pip install librosa kokoro",
        # wan/t5.py touches CUDA at import time; validate xfuser→diffusers path instead.
        f"PYTHONPATH={INFINITETALK_ROOT} python -c \""
        "import torch, xformers, flash_attn, diffusers, transformers, xfuser; "
        "print('deps ok', torch.__version__, diffusers.__version__, transformers.__version__)\"",
        env={"PYTHONPATH": INFINITETALK_ROOT},
    )
    .env({"PYTHONPATH": INFINITETALK_ROOT})
)


@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    timeout=3600,
)
class Inference:
    @modal.enter()
    def load_models(self) -> None:
        import torch
        import wan
        from wan.configs import WAN_CONFIGS

        if not os.path.isfile(LIGHTX2V_LORA_PATH):
            raise RuntimeError(
                f"LightX2V LoRA not found at {LIGHTX2V_LORA_PATH}. "
                "Run download.py (includes Kijai/WanVideo_comfy lightx2v weights)."
            )

        print(
            "[infinitetalk] Loading Wan + InfiniteTalk + lightx2v LoRA (4-step)…",
            flush=True,
        )
        torch.cuda.set_device(0)
        cfg = WAN_CONFIGS["infinitetalk-14B"]
        self._gen = _load_generate_module()
        self._wan_i2v = wan.InfiniteTalkPipeline(
            config=cfg,
            checkpoint_dir=WAN_CKPT_DIR,
            quant_dir=None,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            lora_dir=[LIGHTX2V_LORA_PATH],
            lora_scales=[DEFAULT_LORA_SCALE],
            quant=None,
            dit_path=None,
            infinitetalk_dir=INFINITETALK_WEIGHT,
        )
        self._wav2vec_fe, self._audio_enc = self._gen.custom_init("cpu", WAV2VEC_DIR)
        print("[infinitetalk] Pipeline ready.", flush=True)

    def _generate_talking_video(
        self,
        *,
        reference_bytes: bytes,
        reference_name: str,
        audio_b: bytes,
        text: str | None,
        seed_val: float | None,
    ) -> tuple[bytes | None, str | None]:
        """Shared InfiniteTalk core: (reference + audio) -> talking-head mp4.

        InfiniteTalk's `cond_video` accepts either a still image or a video
        reference, so both slots funnel through here; only `reference_name`
        (file extension) differs. Returns (mp4_bytes, error).
        """
        prompt = (text or "").strip() or DEFAULT_PROMPT
        seed = int(seed_val) if seed_val is not None else 42

        if not os.path.isdir(WAN_CKPT_DIR):
            return None, f"Wan checkpoint not found at {WAN_CKPT_DIR}. Run download.py first."
        if not os.path.isfile(INFINITETALK_WEIGHT):
            return None, (
                f"InfiniteTalk weights not found at {INFINITETALK_WEIGHT}. "
                "Run download.py first."
            )
        if not os.path.isfile(LIGHTX2V_LORA_PATH):
            return None, f"LightX2V LoRA not found at {LIGHTX2V_LORA_PATH}. Run download.py first."

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            reference_path = work / reference_name
            raw_audio_path = work / "input_audio"
            audio_path = work / "target_audio.wav"
            reference_path.write_bytes(reference_bytes)
            raw_audio_path.write_bytes(audio_b)
            try:
                # Use the full speech (no length trim) so the video can track it.
                _normalize_audio_to_wav(raw_audio_path, audio_path)
            except subprocess.CalledProcessError as e:
                err = (e.stderr or e.stdout or str(e))[-2000:]
                return None, f"Audio prepare failed: {err}"

            duration_s = _probe_duration_seconds(str(audio_path))
            frame_num, max_frame_num, mode = _plan_frames(duration_s)

            save_stem = work / "output"
            audio_save_dir = work / "save_audio" / "clip"
            audio_save_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[infinitetalk] Generating: {DEFAULT_SIZE} mode={mode} "
                f"window={frame_num} total_frames={max_frame_num} "
                f"audio={duration_s:.1f}s steps={DEFAULT_SAMPLE_STEPS} lightx2v LoRA",
                flush=True,
            )
            gen = self._gen
            import soundfile as sf
            import torch

            human_speech = gen.audio_prepare_single(str(audio_path))
            audio_embedding = gen.get_embedding(
                human_speech, self._wav2vec_fe, self._audio_enc
            )
            emb_path = audio_save_dir / "1.pt"
            sum_audio = audio_save_dir / "sum.wav"
            sf.write(str(sum_audio), human_speech, 16000)
            torch.save(audio_embedding, str(emb_path))

            manifest = work / "input.json"
            manifest.write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "cond_video": str(reference_path),
                        "cond_audio": {"person1": str(audio_path)},
                        "video_audio": str(sum_audio),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            input_clip = {
                "prompt": prompt,
                "cond_video": str(reference_path),
                "cond_audio": {"person1": str(emb_path)},
                "video_audio": str(sum_audio),
            }
            extra_args = _infinitetalk_cli_args(
                gen,
                size=DEFAULT_SIZE,
                seed=seed,
                frame_num=frame_num,
                max_frame_num=max_frame_num,
                mode=mode,
                input_json=str(manifest),
                audio_save_dir=str(audio_save_dir),
            )

            video_tensor = self._wan_i2v.generate_infinitetalk(
                input_clip,
                size_buckget=DEFAULT_SIZE,
                motion_frame=DEFAULT_MOTION_FRAME,
                frame_num=frame_num,
                shift=DEFAULT_SHIFT,
                sampling_steps=DEFAULT_SAMPLE_STEPS,
                text_guide_scale=DEFAULT_TEXT_GUIDE_SCALE,
                audio_guide_scale=DEFAULT_AUDIO_GUIDE_SCALE,
                seed=seed,
                offload_model=False,
                max_frames_num=max_frame_num,
                color_correction_strength=1.0,
                extra_args=extra_args,
            )
            gen.save_video_ffmpeg(
                video_tensor,
                str(save_stem),
                [str(sum_audio)],
                high_quality_save=False,
            )

            out_mp4 = Path(f"{save_stem}.mp4")
            if not out_mp4.is_file():
                candidates = list(work.glob("**/*.mp4"))
                if not candidates:
                    return None, "InfiniteTalk produced no output mp4"
                out_mp4 = candidates[0]

            return out_mp4.read_bytes(), None

    @modal.method()
    @node_slot(NodeSlots.AUDIO_VIDEO_LIP_SYNC)
    def audio_video_lip_sync(
        self, input: AudioVideoLipSyncInput
    ) -> AudioVideoLipSyncOutput:
        video_b = _maybe_bytes(input.video)
        audio_b = _maybe_bytes(input.audio)
        if not video_b:
            return AudioVideoLipSyncOutput(success=False, error="Missing video")
        if not audio_b:
            return AudioVideoLipSyncOutput(success=False, error="Missing audio")

        mp4, err = self._generate_talking_video(
            reference_bytes=video_b,
            reference_name="reference.mp4",
            audio_b=audio_b,
            text=input.text,
            seed_val=input.seed,
        )
        if err or not mp4:
            return AudioVideoLipSyncOutput(success=False, error=err or "no output")
        return AudioVideoLipSyncOutput(
            success=True,
            video=asset(mp4, mime="video/mp4"),
        )

    @modal.method()
    @node_slot(NodeSlots.AUDIO_IMAGE_GEN_VIDEO)
    def audio_image_gen_video(
        self, input: AudioImageGenVideoInput
    ) -> AudioImageGenVideoOutput:
        image_b = _maybe_bytes(input.image)
        audio_b = _maybe_bytes(input.audio)
        if not image_b:
            return AudioImageGenVideoOutput(success=False, error="Missing image")
        if not audio_b:
            return AudioImageGenVideoOutput(success=False, error="Missing audio")

        # AudioImageGenVideoInput exposes no seed; width/height are advisory and
        # left to the plugin's DEFAULT_SIZE bucket.
        mp4, err = self._generate_talking_video(
            reference_bytes=image_b,
            reference_name="reference.png",
            audio_b=audio_b,
            text=input.text,
            seed_val=None,
        )
        if err or not mp4:
            return AudioImageGenVideoOutput(success=False, error=err or "no output")
        return AudioImageGenVideoOutput(
            success=True,
            video=asset(mp4, mime="video/mp4"),
        )
