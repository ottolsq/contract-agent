import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import embeddings, vectorstore
from .citations import build_aliases, validate_citations
from .config import Settings, get_settings
from .evaluation import load_corpus_articles
from .llm import close_client, stream_chat
from .prompts import build_messages
from .retriever import aretrieve
from .review import review_events
from .schemas import AskRequest, ChatRequest, ReviewRequest
from .vectorstore import Hit

logger = logging.getLogger("contract_agent")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _warm_up() -> None:
    """预加载模型并校验索引；失败直接让启动失败，避免服务带着坏索引对外提供服务。"""
    settings = get_settings()
    started = time.perf_counter()

    embeddings.get_model()
    indexed = vectorstore.count(settings)
    if indexed == 0:
        raise RuntimeError(
            "索引为空：请先运行 uv run python scripts/build_index.py --rebuild"
        )
    vectorstore.verify_model(settings)  # 模型名或维度不匹配会在这里抛出

    _log(
        event="warmup",
        model=embeddings.MODEL_NAME,
        indexed=indexed,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    if get_settings().warmup_on_startup:
        await asyncio.to_thread(_warm_up)
    try:
        yield
    finally:
        await close_client()


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


@lru_cache(maxsize=2)
def _corpus_citation_index(
    path: str, mtime: float
) -> tuple[frozenset[tuple[str, int]], dict[str, str]]:  # noqa: ARG001 mtime 只用于失效
    """语料里真实存在的 (法律名, 条号) 集合，以及短名别名表。

    与评测脚本共用同一份语料，保证接口报的 L1 和 W3 报的 L1 口径一致。
    """
    keys = frozenset(load_corpus_articles(Path(path)))
    return keys, build_aliases(law for law, _ in keys)


def _citation_report(answer: str, settings: Settings) -> dict[str, object] | None:
    """L1 校验：回答里引用的条号是否真实存在于语料。

    只做可观测，不做门禁——正文已经流给用户了，中途拦截会自相矛盾。
    门禁交给评测脚本（`--fail-under-l1`）。
    """
    path = Path(settings.laws_jsonl)
    if not answer.strip() or not path.exists():
        return None

    keys, aliases = _corpus_citation_index(str(path), path.stat().st_mtime)
    report = validate_citations(answer, keys, aliases)

    payload: dict[str, object] = {
        "total": report.total,
        "valid": report.valid,
        "invalid": report.invalid,
        "unresolved": report.unresolved,
        "accuracy": None if report.accuracy is None else round(report.accuracy, 4),
        "has_valid": report.has_valid,
    }
    invalid_refs = [
        {"law": citation.law or citation.law_raw, "article_no": citation.article_no}
        for citation in report.resolved
        if not citation.valid
    ]
    if invalid_refs:
        payload["invalid_refs"] = invalid_refs
    return payload


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

    # 空结果要区分"索引没建"和"确实没匹配到"，否则会误判成法条不覆盖
    if not hits and vectorstore.count(settings) == 0:
        _log(event="index_empty", trace_id=trace_id)
        yield _sse({"error": "知识库索引为空，请先运行 scripts/build_index.py --rebuild"})
        return

    yield _sse({"references": _references(hits)})

    messages = build_messages(payload.question, hits, payload.contract_type)
    chunks: list[str] = []
    citation_total: int | None = None
    try:
        async for token in stream_chat(settings, messages):
            chunks.append(token)
            yield _sse({"delta": token})
    except Exception as exc:
        _log(
            event="generate_failed",
            trace_id=trace_id,
            error=type(exc).__name__,
        )
        yield _sse({"error": "模型服务调用失败，请稍后重试"})
    else:
        # 校验失败不能拖垮已经完成的回答：降级成"没有引用信息"，正文照常交付
        try:
            report = _citation_report("".join(chunks), settings)
        except Exception as exc:
            _log(
                event="citations_failed",
                trace_id=trace_id,
                error=type(exc).__name__,
            )
            report = None

        if report is not None:
            citation_total = int(report["total"])  # type: ignore[arg-type]
            yield _sse({"citations": report})
            _log(
                event="citations",
                trace_id=trace_id,
                total=report["total"],
                valid=report["valid"],
                invalid=report["invalid"],
                accuracy=report["accuracy"],
            )
        yield _sse("[DONE]")
    finally:
        _log(
            event="ask_done",
            trace_id=trace_id,
            total_ms=round((time.perf_counter() - started) * 1000),
            references=len(hits),
            citations=citation_total,
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


async def _review_sse(settings: Settings, payload: ReviewRequest):
    """把编排事件翻译成 SSE：stage / clause / report，最后 [DONE]。"""
    trace_id = uuid4().hex[:12]
    started = time.perf_counter()
    try:
        async for event in review_events(
            payload.contract_text,
            settings=settings,
            contract_type=payload.contract_type,
        ):
            if event.kind == "report":
                report = event.payload["report"]
                _log(
                    event="review_done",
                    trace_id=trace_id,
                    status=report["status"],
                    clauses=len(report["clauses"]),
                    failures=len(report["failures"]),
                    total_ms=round((time.perf_counter() - started) * 1000),
                )
                yield _sse({"report": report})
            elif event.kind == "clause":
                yield _sse({"clause": event.payload})
            else:
                yield _sse({"stage": event.payload})
        yield _sse("[DONE]")
    except ValueError as exc:
        yield _sse({"error": str(exc)})
    except Exception as exc:
        # 不把上游细节透传出去，避免泄露服务信息或密钥
        _log(event="review_failed", trace_id=trace_id, error=type(exc).__name__)
        yield _sse({"error": "审查失败，请稍后重试"})


@app.post("/api/review/stream")
async def review_stream(payload: ReviewRequest) -> StreamingResponse:
    settings = get_settings()
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY 未配置")

    return StreamingResponse(
        _review_sse(settings, payload),
        media_type="text/event-stream",
    )
