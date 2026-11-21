from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，全部来自环境变量或 .env 文件。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_timeout: float = 60.0
    # 输出侧上限；不设会让长报告无界生成
    llm_max_tokens: int = 4096
    # 仅重试瞬时故障（429 / 5xx / 连接错误）；流式已输出内容时不回退重试
    llm_max_retries: int = 3
    # 向量库持久化目录，索引属于构建产物，不入库
    chroma_dir: str = "data/chroma"
    # 启动时预加载 embedding 模型：线上建议开启，避免首个请求等十几秒；
    # 用 --reload 开发时可设为 false
    warmup_on_startup: bool = True

    # 检索模式：hybrid = 向量 + BM25 融合；dense 只走向量；bm25 只走词面（不加载模型）
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    # 融合权重：给 BM25 的比重。0 = 纯 dense，1 = 纯 BM25；
    # 0.7 是在 W3 评测集上做留一法交叉验证选出来的（30 折全部选中 0.7）
    hybrid_bm25_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    # 融合前向量侧多取的候选数；BM25 侧默认覆盖域内全部法条，无需单设
    hybrid_candidates: int = Field(default=100, ge=1)
    # 法条语料路径：BM25 索引与评测都从这里读，与 build_index 的输入保持一致
    laws_jsonl: str = "data/processed/laws.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
