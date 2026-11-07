from functools import lru_cache

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
    # 仅重试瞬时故障（429 / 5xx / 连接错误）；流式已输出内容时不回退重试
    llm_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
