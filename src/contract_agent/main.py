import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import Settings, get_settings
from .llm import stream_chat
from .schemas import ChatRequest

app = FastAPI(title="Contract Agent", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "model": settings.llm_model,
        "api_key_configured": bool(settings.llm_api_key),
    }


def _sse(payload: object) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _chat_events(settings: Settings, messages: list[dict[str, str]]):
    try:
        async for token in stream_chat(settings, messages):
            yield _sse({"delta": token})
        yield _sse("[DONE]")
    except Exception:
        # 不把上游错误细节暴露给客户端，避免泄露密钥或服务信息
        yield _sse({"error": "模型服务调用失败，请稍后重试"})


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    settings = get_settings()
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY 未配置")

    messages = [message.model_dump() for message in payload.messages]
    return StreamingResponse(
        _chat_events(settings, messages),
        media_type="text/event-stream",
    )
