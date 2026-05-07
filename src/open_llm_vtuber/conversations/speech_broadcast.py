import asyncio
import json
from typing import Optional

from loguru import logger

from ..agent.output_types import Actions, DisplayText
from ..service_context import ServiceContext
from .conversation_utils import (
    cleanup_conversation,
    send_conversation_end_signal,
)
from .speech_refiner import EMOTION_MAP, refine_speech_segments
from .tts_manager import TTSTaskManager
from .types import WebSocketSend


async def process_speech_broadcast(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    text: str,
    tts_text: Optional[str] = None,
) -> str:
    """Speak externally provided text without invoking ASR, LLM, tools, or history."""
    tts_manager = TTSTaskManager()

    try:
        text = text.strip()
        if not text:
            await websocket_send(
                json.dumps({"type": "error", "message": "Text to speak is empty"})
            )
            return ""

        segments = await refine_speech_segments(context, text)
        full_display_text = "".join(segment.tagged_text for segment in segments)

        await websocket_send(
            json.dumps({"type": "control", "text": "conversation-chain-start"})
        )
        await websocket_send(
            json.dumps({"type": "full-text", "text": full_display_text})
        )
        logger.info(f"Broadcasting external speech: {text}")
        logger.info(
            "Refined speech segments: "
            + ", ".join(
                f"{segment.action}->{EMOTION_MAP[segment.action]}"
                for segment in segments
            )
        )

        for segment in segments:
            display_text = segment.tagged_text
            expression_name = EMOTION_MAP.get(segment.action)
            actions = Actions(expressions=[expression_name] if expression_name else None)
            if tts_text is not None and len(segments) == 1:
                speech_text = tts_text
            elif getattr(context.tts_engine, "uses_emotion_instructions", False):
                speech_text = display_text
            else:
                speech_text = context.live2d_model.remove_emotion_keywords(display_text)
            speech_text = speech_text.strip()

            await tts_manager.speak(
                tts_text=speech_text,
                display_text=DisplayText(
                    text=display_text,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                ),
                actions=actions,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=websocket_send,
                instruct=segment.instruct,
            )

        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)

        await tts_manager.wait_until_payloads_sent()
        await websocket_send(json.dumps({"type": "backend-synth-complete"}))
        await send_conversation_end_signal(
            websocket_send=websocket_send,
            broadcast_ctx=None,
            session_emoji="speech-broadcast",
        )
        return text

    except asyncio.CancelledError:
        logger.info("Speech broadcast cancelled.")
        raise
    except Exception as e:
        logger.error(f"Error in speech broadcast: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Speech broadcast error: {e}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, "speech-broadcast")
