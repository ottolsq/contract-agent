from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    contract_type: Literal["internal", "external", "all"] = "all"
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能为空或仅含空白字符")
        return value
