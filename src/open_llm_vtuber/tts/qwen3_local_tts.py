import os
import platform
import re
from typing import Literal

import soundfile as sf
from loguru import logger

from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """Local Qwen3-TTS engine using the qwen-tts Python package."""

    uses_emotion_instructions = True

    _emotion_to_expression = {
        "neutral": "f01",
        "anger": "f03",
        "disgust": "f03",
        "fear": "f02",
        "joy": "f04",
        "smirk": "f04",
        "sadness": "f02",
        "surprise": "f04",
    }

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        mode: Literal["custom_voice", "voice_design"] = "custom_voice",
        language: str = "Chinese",
        speaker: str = "Vivian",
        instruct: str = "",
        device_map: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "",
    ) -> None:
        self.model_path = self._ensure_str(model_path)
        self.mode = self._ensure_str(mode)
        self.language = self._ensure_str(language)
        self.speaker = self._ensure_str(speaker)
        self.instruct = self._ensure_str(instruct)
        self.device_map = self._ensure_str(device_map)
        self.dtype = self._ensure_str(dtype)
        self.attn_implementation = self._ensure_str(attn_implementation)
        self.file_extension = "wav"
        self._model = None

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        text = self._ensure_str(text)
        emotions = self._extract_emotions(text)
        text = self._normalize_text(self._remove_emotion_tags(text))
        if not text:
            return None

        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        try:
            model = self._load_model()
            instruct = self._build_instruct(emotions)

            if self.mode == "voice_design":
                wavs, sample_rate = model.generate_voice_design(
                    text=text,
                    language=self.language,
                    instruct=instruct or "",
                )
            elif self.mode == "custom_voice":
                wavs, sample_rate = model.generate_custom_voice(
                    text=text,
                    language=self.language,
                    speaker=self.speaker,
                    instruct=instruct or "",
                )
            else:
                raise ValueError(f"Unsupported Qwen3 local TTS mode: {self.mode}")

            if not wavs:
                raise RuntimeError("Qwen3 local TTS returned no audio")

            sf.write(file_name, wavs[0], sample_rate)
            return file_name

        except Exception as e:
            logger.exception(f"Exception in qwen3_local_tts generate_audio: {e}")
            if os.path.exists(file_name):
                os.remove(file_name)
            return None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "qwen-tts is required for qwen3_local_tts. "
                "Install it with: uv sync --extra qwen3-tts"
            ) from exc

        kwargs = {"device_map": self._resolve_device_map(torch)}
        dtype = self._resolve_torch_dtype(torch)
        if dtype is not None:
            kwargs["dtype"] = dtype
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        logger.info(f"Loading local Qwen3-TTS model: {self.model_path}")
        self._model = Qwen3TTSModel.from_pretrained(self.model_path, **kwargs)
        return self._model

    def _resolve_device_map(self, torch) -> str:
        if self.device_map != "auto":
            return self.device_map

        if platform.system() == "Darwin":
            logger.info(
                "Using CPU for Qwen3 local TTS on macOS. "
                "device_map=auto can offload tensors to meta/disk, while MPS "
                "does not support Qwen3-TTS decoder Conv1d output channels."
            )
            return "cpu"

        return self.device_map

    def _resolve_torch_dtype(self, torch):
        if not self.dtype or self.dtype == "auto":
            return None
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.dtype not in dtype_map:
            raise ValueError(
                "Unsupported Qwen3 local TTS dtype "
                f"{self.dtype!r}; use auto, float16, bfloat16, or float32"
            )
        return dtype_map[self.dtype]

    def _build_instruct(self, emotions: list[str]) -> str:
        emotion_hint = self._build_emotion_hint(emotions)
        if self.instruct and emotion_hint:
            return f"{self.instruct}\n{emotion_hint}"
        return self.instruct or emotion_hint

    def _build_emotion_hint(self, emotions: list[str]) -> str:
        if not emotions:
            return ""

        emotion_codes = [
            f"{emotion}={self._emotion_to_expression[emotion]}" for emotion in emotions
        ]
        return (
            "本次播报文本包含情绪标签："
            + "、".join(emotion_codes)
            + "。请根据这些情绪调整语气、能量和节奏；不要读出情绪标签或表情编号。"
        )

    def _extract_emotions(self, text: str) -> list[str]:
        text = self._ensure_str(text)
        found = []
        lower_text = text.lower()
        for emotion in self._emotion_to_expression:
            if f"[{emotion}]" in lower_text and emotion not in found:
                found.append(emotion)
        return found

    def _remove_emotion_tags(self, text: str) -> str:
        text = self._ensure_str(text)
        pattern = "|".join(
            re.escape(f"[{emotion}]") for emotion in self._emotion_to_expression
        )
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    @staticmethod
    def _ensure_str(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.strip().replace("\n", "，")
        max_len = 900
        if len(text) > max_len:
            text = text[:max_len] + "，先说到这里"
        return text
