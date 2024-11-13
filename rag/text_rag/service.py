import time

from generator.llm_generator import LLMGenerator
from retriever.text_similarity_retriever import TextSimilarityRetriever
from text_rag.schema import Output


class TextRAG:
    """Text RAG with retriever based on embeddings similarity and an LLM as generator."""

    def __init__(self) -> None:
        self.retriever = TextSimilarityRetriever()
        self.generator = LLMGenerator()

    def _format_output(
        self,
        execution_time: float,
        context: str,
        answer: str,
        env_impacts: list[dict[str, any]],
    ) -> Output:
        """Change the output format."""

        output = Output(
            execution_time=execution_time,
            context=context,
            answer=answer,
            env_impacts=env_impacts,
        )
        return output

    def _format_env_impact(self, generator_output) -> list[dict[str, any]]:
        """Format environmental impact scores from the generator."""
        env_impacts = []
        env_impacts.append(
            {
                "name": generator_output["env_impacts"].energy.name,
                "value": generator_output["env_impacts"].energy.value,
                "unit": generator_output["env_impacts"].energy.unit,
            }
        )
        env_impacts.append(
            {
                "name": generator_output["env_impacts"].gwp.name,
                "value": generator_output["env_impacts"].gwp.value,
                "unit": generator_output["env_impacts"].gwp.unit,
            }
        )
        env_impacts.append(
            {
                "name": generator_output["env_impacts"].adpe.name,
                "value": generator_output["env_impacts"].adpe.value,
                "unit": generator_output["env_impacts"].adpe.unit,
            }
        )
        env_impacts.append(
            {
                "name": generator_output["env_impacts"].pe.name,
                "value": generator_output["env_impacts"].pe.value,
                "unit": generator_output["env_impacts"].pe.unit,
            }
        )
        return env_impacts

    def run(self, question: str) -> Output:
        """Answer a question based on a knowledge graph."""
        start_time = time.time()
        retriever_output = self.retriever.run(question)
        generator_output = self.generator.run(retriever_output["context"], question)
        execution_time = time.time() - start_time
        formatted_output = self._format_output(
            execution_time,
            retriever_output["context"],
            generator_output["answer"],
            self._format_env_impact(generator_output),
        )
        return formatted_output
