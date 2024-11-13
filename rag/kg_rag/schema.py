from pydantic import BaseModel


class Output(BaseModel):
    """Output format for KG-RAG."""

    execution_time: float
    entities: list[str]
    queries: list[str]
    llm_query: str
    context: str
    answer: str
    env_impacts: list[dict[str, str | float]]
