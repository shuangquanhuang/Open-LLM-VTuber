import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger
from openai import OpenAI


EMOTION_MAP = {
    "neutral": "f01",
    "anger": "f03",
    "disgust": "f03",
    "fear": "f02",
    "joy": "f04",
    "smirk": "f04",
    "sadness": "f02",
    "surprise": "f04",
}


@dataclass
class SpeechSegment:
    text: str
    instruct: str
    action: str

    @property
    def tagged_text(self) -> str:
        if self.action in EMOTION_MAP and f"[{self.action}]" not in self.text.lower():
            return f"{self.text}[{self.action}]"
        return self.text


async def refine_speech_segments(context, text: str) -> list[SpeechSegment]:
    """Rewrite externally supplied text into VTuber-ready speech segments."""
    return await asyncio.to_thread(_refine_speech_segments_sync, context, text)


def _refine_speech_segments_sync(context, text: str) -> list[SpeechSegment]:
    api_key = _resolve_deepseek_key(context)
    if not api_key:
        logger.warning("DeepSeek API key is not configured; using original text.")
        return _fallback_segments(text)

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=_resolve_deepseek_base_url(context),
            timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "15")),
        )
        response = client.chat.completions.create(
            model=_resolve_deepseek_model(context),
            temperature=0.7,
            max_tokens=900,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _build_system_prompt(context)},
                {"role": "user", "content": text},
            ],
        )
        content = response.choices[0].message.content or ""
        logger.info(f"DeepSeek refined speech raw output: {content}")
        segments = _parse_segments(content)
        if segments:
            logger.info(f"DeepSeek refined speech into {len(segments)} segment(s).")
            for index, segment in enumerate(segments, start=1):
                logger.info(
                    "DeepSeek refined speech segment "
                    f"{index}: text={segment.text!r}, "
                    f"instruct={segment.instruct!r}, action={segment.action!r}, "
                    f"tagged_text={segment.tagged_text!r}"
                )
            return segments
        logger.warning("DeepSeek returned no usable speech segments; using original text.")
    except Exception as exc:
        logger.error(f"DeepSeek speech refinement failed: {exc}")

    return _fallback_segments(text)


def _build_system_prompt(context) -> str:
    character_name = getattr(context.character_config, "character_name", "VTuber")
    persona_prompt = getattr(context.character_config, "persona_prompt", "")
    return f"""
你是 Open-LLM-VTuber 的播报改写器。你只输出 JSON，不要输出 Markdown。

角色名：{character_name}
角色人设：
{persona_prompt}

任务：
把用户提供的文本润色成更适合该 Live2D 角色播报的短句，但不要改变事实含义。
输出 JSON 对象，格式必须是：
{{
  "segments": [
    {{
      "text": "实际要展示和播报的中文短句，不要包含 markdown",
      "instruct": "给 TTS 的语气说明，描述音色、语速、情绪、停顿，不要要求读出情绪标签",
      "action": "neutral"
    }}
  ]
}}

要求：
- segments 数量 1 到 4 段，每段尽量 12 到 60 个中文字符。
- action 只能从这些键里选择：{", ".join(EMOTION_MAP.keys())}
- text 中可以自然加入 [action] 情绪标签，或不加；系统会自动补充。
- instruct 必须与该段 action 匹配，比如 joy 更明亮，anger 更有压迫感，sadness 更低落。
- 不要在 text 中读出 f01/f02/f03/f04，也不要解释 JSON 格式。
- 保留原始文本中的关键信息、称呼、数字、专有名词。
""".strip()


def _parse_segments(content: str) -> list[SpeechSegment]:
    data = _json_loads_lenient(content)
    raw_segments = data.get("segments", []) if isinstance(data, dict) else []
    segments = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        instruct = str(raw.get("instruct", "")).strip()
        action = str(raw.get("action", "neutral")).strip().lower()
        if not text:
            continue
        if action not in EMOTION_MAP:
            action = "neutral"
        segments.append(SpeechSegment(text=text, instruct=instruct, action=action))
    return segments[:4]


def _json_loads_lenient(content: str) -> Any:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def _fallback_segments(text: str) -> list[SpeechSegment]:
    action = _extract_first_emotion(text) or "neutral"
    return [
        SpeechSegment(
            text=text.strip(),
            instruct="自然、清亮、贴近角色人设地播报；不要读出情绪标签。",
            action=action,
        )
    ]


def _extract_first_emotion(text: str) -> str | None:
    lower_text = text.lower()
    for emotion in EMOTION_MAP:
        if f"[{emotion}]" in lower_text:
            return emotion
    return None


def _resolve_deepseek_key(context) -> str:
    env_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    deepseek_config = _get_deepseek_config(context)
    api_key = getattr(deepseek_config, "llm_api_key", "") if deepseek_config else ""
    if api_key and "your deepseek" not in api_key.lower():
        return api_key.strip()
    return ""


def _resolve_deepseek_base_url(context) -> str:
    deepseek_config = _get_deepseek_config(context)
    return (
        getattr(deepseek_config, "base_url", "https://api.deepseek.com/v1")
        or "https://api.deepseek.com/v1"
    )


def _resolve_deepseek_model(context) -> str:
    deepseek_config = _get_deepseek_config(context)
    return getattr(deepseek_config, "model", "deepseek-chat") or "deepseek-chat"


def _get_deepseek_config(context):
    try:
        return context.character_config.agent_config.llm_configs.deepseek_llm
    except Exception:
        return None
