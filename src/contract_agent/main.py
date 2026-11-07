import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import embeddings, vectorstore
from .config import Settings, get_settings
from .llm import stream_chat
from .prompts import build_messages
from .retriever import aretrieve
from .schemas import AskRequest, ChatRequest
from .vectorstore import Hit

logger = logging.getLogger("contract_agent")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _warm_up() -> None:
    """预加载模型并连一次向量库，避免首个请求等十几秒。"""
    settings = get_settings()
    started = time.perf_counter()
    try:
        embeddings.get_model()
        indexed = vectorstore.count(settings)
        if indexed:
            vectorstore.verify_model(settings)
        _log(
            event="warmup",
            model=embeddings.MODEL_NAME,
            indexed=indexed,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        # 预热失败不阻塞启动；真正的错误会在请求时以更明确的错误返回
        _log(event="warmup_failed", error=type(exc).__name__, detail=str(exc)[:200])


@asynccontextmanager
async def lifespan(_: FastAPI):
    if get_settings().warmup_on_startup:
        await asyncio.to_thread(_warm_up)
    yield


app = FastAPI(title="Contract Agent", version="0.2.0", lifespan=lifespan)


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


def _log(**fields: object) -> None:
    logger.info(json.dumps(fields, ensure_ascii=False))


def _references(hits: list[Hit]) -> list[dict[str, object]]:
    return [
        {
            "law": hit.law,
            "article": hit.article,
            "article_no": hit.article_no,
            "domain": hit.domain,
            "similarity": hit.similarity,
            "text": hit.text[:200],
        }
        for hit in hits
    ]


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


async def _ask_events(settings: Settings, payload: AskRequest):
    trace_id = uuid4().hex[:12]
    started = time.perf_counter()

    try:
        hits = await aretrieve(
            payload.question, payload.contract_type, payload.top_k, settings
        )
    except Exception as exc:
        _log(
            event="retrieve_failed",
            trace_id=trace_id,
            contract_type=payload.contract_type,
            error=type(exc).__name__,
        )
        yield _sse({"error": "检索失败，请稍后重试"})
        return

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    _log(
        event="retrieve",
        trace_id=trace_id,
        contract_type=payload.contract_type,
        top_k=payload.top_k,
        hits=[f"{hit.law}#{hit.article_no}" for hit in hits],
        top_similarity=hits[0].similarity if hits else None,
        elapsed_ms=elapsed_ms,
    )

    yield _sse({"references": _references(hits)})

    messages = build_messages(payload.question, hits, payload.contract_type)
    try:
        async for token in stream_chat(settings, messages):
            yield _sse({"delta": token})
        yield _sse("[DONE]")
    except Exception as exc:
        _log(
            event="generate_failed",
            trace_id=trace_id,
            error=type(exc).__name__,
        )
        yield _sse({"error": "模型服务调用失败，请稍后重试"})
    finally:
        _log(
            event="ask_done",
            trace_id=trace_id,
            total_ms=round((time.perf_counter() - started) * 1000),
            references=len(hits),
        )


@app.post("/api/ask/stream")
async def ask_stream(payload: AskRequest) -> StreamingResponse:
    settings = get_settings()
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY 未配置")

    return StreamingResponse(
        _ask_events(settings, payload),
        media_type="text/event-stream",
    )
