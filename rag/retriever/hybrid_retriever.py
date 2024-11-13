from retriever.el_retriever import ELRetriever
from retriever.llm_retriever import LLMRetriever


class HybridRetriever:
    """KG retriever mixing entity linking and query patterns with LLM for the RAG task."""

    def __init__(self) -> None:
        self.el_retriever = ELRetriever()
        self.llm_retriever = LLMRetriever()

    def run(self, question: str) -> dict[str, any]:
        """Retrieve context for a question based on a graph."""
        output = self.el_retriever.run(question)
        if output["context"] == "[]":
            llm_output = self.llm_retriever.run(question)
            output["context"] = llm_output["context"]
            output["llm_query"] = llm_output["query"]
            output["env_impacts"] = llm_output["env_impacts"]
        return output
