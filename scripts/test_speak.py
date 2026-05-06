import argparse
import json
import time

import websocket


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a test speech message to Open LLM VTuber."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="这是一条来自 Hermes Agent 的测试播报。收到后请直接说出来。",
        help="Text to display and speak.",
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:12393/client-ws",
        help="Open LLM VTuber WebSocket URL.",
    )
    parser.add_argument(
        "--tts-text",
        default=None,
        help="Optional text sent to TTS when different from the displayed text.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Seconds to wait for playback completion before exiting.",
    )
    args = parser.parse_args()

    ws = websocket.create_connection(args.url, timeout=10)
    ws.settimeout(1)

    payload = {"type": "speak", "text": args.text}
    if args.tts_text:
        payload["tts_text"] = args.tts_text

    print(f"Connected to {args.url}")
    print(f"Sending: {json.dumps(payload, ensure_ascii=False)}")
    ws.send(json.dumps(payload, ensure_ascii=False))

    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            try:
                message = json.loads(ws.recv())
            except TimeoutError:
                continue
            except websocket.WebSocketTimeoutException:
                continue

            print(json.dumps(message, ensure_ascii=False))

            if (
                message.get("type") == "control"
                and message.get("text") == "conversation-chain-end"
            ):
                break
    finally:
        ws.close()


if __name__ == "__main__":
    main()
