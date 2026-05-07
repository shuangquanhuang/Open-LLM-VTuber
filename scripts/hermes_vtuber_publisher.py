import argparse
import json
import os
import re
import time
from collections.abc import Iterable

import websocket


DEFAULT_WS_URL = "ws://127.0.0.1:12393/client-ws"
SENTENCE_END_RE = re.compile(r"(.+?[。！？!?；;]+)(.*)", re.S)


def publish_vtuber_speech(
    text: str,
    *,
    url: str | None = None,
    tts_text: str | None = None,
    wait: bool = False,
    timeout: float = 300,
) -> None:
    """Push one utterance to Open-LLM-VTuber for playback."""
    text = str(text or "").strip()
    if not text:
        return

    url = url or os.getenv("OPEN_LLM_VTUBER_WS_URL", DEFAULT_WS_URL)
    ws = websocket.create_connection(url, timeout=10)
    ws.settimeout(1)

    payload = {"type": "speak", "text": text}
    if tts_text:
        payload["tts_text"] = str(tts_text)

    try:
        ws.send(json.dumps(payload, ensure_ascii=False))
        if wait:
            _wait_for_playback_complete(ws, timeout)
    finally:
        ws.close()


def publish_streaming_chunks(
    chunks: Iterable[str],
    *,
    url: str | None = None,
    min_chars: int = 18,
) -> str:
    """
    Publish generated text while Hermes is streaming.

    Complete sentences are sent to VTuber as soon as punctuation arrives. The
    full response is returned so the caller can still send it back to Open WebUI.
    """
    full_text = ""
    pending = ""

    for chunk in chunks:
        chunk = str(chunk or "")
        if not chunk:
            continue

        full_text += chunk
        pending += chunk

        while True:
            match = SENTENCE_END_RE.match(pending)
            if not match:
                break

            sentence, pending = match.groups()
            if len(sentence.strip()) >= min_chars:
                publish_vtuber_speech(sentence, url=url)
            else:
                pending = sentence + pending
                break

    if pending.strip():
        publish_vtuber_speech(pending, url=url)

    return full_text


def _wait_for_playback_complete(ws, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            message = json.loads(ws.recv())
        except TimeoutError:
            continue
        except websocket.WebSocketTimeoutException:
            continue

        if (
            message.get("type") == "control"
            and message.get("text") == "conversation-chain-end"
        ):
            return

    raise TimeoutError("Timed out waiting for VTuber playback completion")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish Hermes/Open WebUI text to Open-LLM-VTuber."
    )
    parser.add_argument("text", help="Text to display and speak.")
    parser.add_argument(
        "--url",
        default=os.getenv("OPEN_LLM_VTUBER_WS_URL", DEFAULT_WS_URL),
        help="Open-LLM-VTuber WebSocket URL.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until Open-LLM-VTuber reports playback completion.",
    )
    args = parser.parse_args()

    publish_vtuber_speech(args.text, url=args.url, wait=args.wait)


if __name__ == "__main__":
    main()
