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

        expressions = context.live2d_model.extract_emotion(text)
        actions = Actions(expressions=expressions or None)
        display_text = text
        speech_text = (
            tts_text
            if tts_text is not None
            else context.live2d_model.remove_emotion_keywords(text)
        ).strip()

        await websocket_send(
            json.dumps({"type": "control", "text": "conversation-chain-start"})
        )
        await websocket_send(json.dumps({"type": "full-text", "text": "Speaking..."}))
        logger.info(f"Broadcasting external speech: {text}")
        if expressions:
            logger.info(f"Extracted Live2D expressions: {expressions}")

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
