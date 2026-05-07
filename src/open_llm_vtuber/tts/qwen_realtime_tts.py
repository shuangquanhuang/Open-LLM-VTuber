import base64
import json
import os
import re
import threading
import time
import wave
from typing import Optional

import dashscope
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from loguru import logger

from .tts_interface import TTSInterface


class _WavCollectCallback(QwenTtsRealtimeCallback):
    """Collect realtime PCM chunks returned by DashScope Qwen TTS."""

    def __init__(self) -> None:
        super().__init__()
        self.complete_event = threading.Event()
        self.audio_chunks: list[bytes] = []
        self.error: Optional[str] = None
        self.last_event_type: Optional[str] = None
        self.last_audio_at: Optional[float] = None

    def on_open(self) -> None:
        logger.debug("Qwen realtime TTS connection opened")

    def on_close(self, close_status_code, close_msg) -> None:
        logger.debug(
            f"Qwen realtime TTS connection closed, code={close_status_code}, msg={close_msg}"
        )
        if self.audio_chunks:
            self.complete_event.set()

    def on_event(self, response: dict | str) -> None:
        try:
            if isinstance(response, str):
                response = json.loads(response)

            event_type = response.get("type")
            self.last_event_type = event_type

            if event_type == "response.audio.delta":
                self.audio_chunks.append(base64.b64decode(response["delta"]))
                self.last_audio_at = time.monotonic()
            elif event_type == "response.done":
                self.complete_event.set()
            elif event_type == "session.finished":
                self.complete_event.set()
            elif event_type == "error":
                self.error = str(response)
                logger.error(f"Qwen realtime TTS error: {response}")
                self.complete_event.set()

        except Exception as exc:
            self.error = str(exc)
            logger.error(f"Qwen realtime TTS callback error: {exc}")
            self.complete_event.set()

    def wait_for_finished(self, timeout: Optional[float] = None) -> bool:
        return self.complete_event.wait(timeout=timeout)

    def wait_for_audio_complete(
        self,
        timeout: float,
        idle_after_audio: float,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.complete_event.wait(timeout=0.1):
                return True
            if (
                self.audio_chunks
                and self.last_audio_at is not None
                and time.monotonic() - self.last_audio_at >= idle_after_audio
            ):
                logger.warning(
                    "Qwen realtime TTS did not emit a completion event; "
                    "using collected audio after idle timeout."
                )
                return True
        return False


class TTSEngine(TTSInterface):
    """DashScope Qwen realtime TTS engine that writes collected PCM as WAV."""

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
        model: str = "qwen3-tts-instruct-flash-realtime",
        voice: str = "Bunny",
        region: str = "cn",
        api_key: str = "",
        language_type: str = "Chinese",
        instructions: str = "",
        optimize_instructions: bool = True,
        speech_rate: float = 1.08,
        pitch_rate: float = 1.05,
        volume: int = 70,
        timeout: float = 30,
        idle_after_audio: float = 2,
    ) -> None:
        self.model = model
        self.voice = voice
        self.region = region
        self.language_type = language_type
        self.instructions = instructions
        self.optimize_instructions = optimize_instructions
        self.speech_rate = speech_rate
        self.pitch_rate = pitch_rate
        self.volume = volume
        self.timeout = timeout
        self.idle_after_audio = idle_after_audio
        self.sample_rate = 24000
        self.file_extension = "wav"

        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not resolved_api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY is required for qwen_realtime_tts."
            )
        dashscope.api_key = resolved_api_key

    @staticmethod
    def _get_url(region: str) -> str:
        if region.lower() in {"intl", "sg", "singapore"}:
            return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
        return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    def generate_audio(
        self, text: str, file_name_no_ext=None, instruct: str | None = None
    ) -> str:
        emotions = self._extract_emotions(text)
        text = self._normalize_text(self._remove_emotion_tags(text))
        if not text:
            return None

        file_name = self.generate_cache_file_name(
            file_name_no_ext, self.file_extension
        )
        callback = _WavCollectCallback()
        tts = QwenTtsRealtime(
            model=self.model,
            callback=callback,
            url=self._get_url(self.region),
        )

        try:
            tts.connect()
            tts.update_session(
                voice=self.voice,
                response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                mode="server_commit",
                language_type=self.language_type,
                instructions=self._build_instructions(emotions, instruct=instruct),
                optimize_instructions=self.optimize_instructions,
                speech_rate=self.speech_rate,
                pitch_rate=self.pitch_rate,
                volume=self.volume,
            )
            tts.append_text(text)
            tts.finish()

            if not callback.wait_for_audio_complete(
                timeout=self.timeout,
                idle_after_audio=self.idle_after_audio,
            ):
                raise TimeoutError(
                    "Timed out waiting for Qwen realtime TTS "
                    f"(last_event={callback.last_event_type}, "
                    f"audio_chunks={len(callback.audio_chunks)})"
                )
            if callback.error:
                raise RuntimeError(callback.error)
            if not callback.audio_chunks:
                raise RuntimeError("Qwen realtime TTS returned no audio")

            self._write_wav(file_name, b"".join(callback.audio_chunks))
            return file_name

        except Exception as e:
            logger.error(f"Exception in qwen_realtime_tts generate_audio: {e}")
            if os.path.exists(file_name):
                os.remove(file_name)
            return None
        finally:
            try:
                tts.close()
            except Exception as e:
                logger.debug(f"Error closing Qwen realtime TTS connection: {e}")

    def _write_wav(self, file_name: str, pcm_bytes: bytes) -> None:
        with wave.open(file_name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_bytes)

    def _build_instructions(
        self, emotions: list[str], instruct: str | None = None
    ) -> str:
        instruction_parts = [part for part in [self.instructions, instruct] if part]

        if emotions:
            emotion_codes = [
                f"{emotion}={self._emotion_to_expression[emotion]}"
                for emotion in emotions
            ]
            instruction_parts.append(
                "本次播报文本包含情绪标签："
                + "、".join(emotion_codes)
                + "。请根据这些情绪调整语气、能量和节奏；不要读出情绪标签或表情编号。"
            )

        return "\n".join(instruction_parts)

    def _extract_emotions(self, text: str) -> list[str]:
        found = []
        lower_text = text.lower()
        for emotion in self._emotion_to_expression:
            if f"[{emotion}]" in lower_text and emotion not in found:
                found.append(emotion)
        return found

    def _remove_emotion_tags(self, text: str) -> str:
        pattern = "|".join(
            re.escape(f"[{emotion}]") for emotion in self._emotion_to_expression
        )
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.strip().replace("\n", "，")
        max_len = 900
        if len(text) > max_len:
            text = text[:max_len] + "，先说到这里"
        return text
