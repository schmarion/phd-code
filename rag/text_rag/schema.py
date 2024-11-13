from pydantic import BaseModel


class Output(BaseModel):
    """Output format for Text-RAG."""

    execution_time: float
    context: str
    answer: str
    env_impacts: list[dict[str, str | float]]
