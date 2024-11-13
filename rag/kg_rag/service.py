import time

from generator.llm_generator import LLMGenerator
from retriever.hybrid_retriever import HybridRetriever
from kg_rag.schema import Output


class KGRAG:
    """KG RAG with hybrid retriever (mix between entity linking,
    query patterns and LLM) and an LLM as generator."""

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.generator = LLMGenerator()

    def _format_output(
        self,
        execution_time: float,
        entities: list[str],
        queries: list[str],
        llm_query: str,
        context: str,
        answer: str,
        env_impacts: list[dict[str, any]],
    ) -> Output:
        """Change the output format."""

        output = Output(
            execution_time=execution_time,
            entities=entities,
            queries=queries,
            llm_query=llm_query,
            context=context,
            answer=answer,
            env_impacts=env_impacts,
        )
        return output

    def _merge_env_impact(
        self, retriever_output, generator_output
    ) -> list[dict[str, any]]:
        """Merge environmental impact scores from the retriever and the generator."""
        env_impacts = []
        env_impacts.append(
            {
                "name": generator_output["env_impacts"].energy.name,
                "value": generator_output["env_impacts"].energy.value
                + (
                    retriever_output["env_impacts"].energy.value
                    if "env_impacts" in retriever_output.keys()
                    else 0
                ),
                "unit": generator_output["env_impacts"].energy.unit,
            }
        )
        env_impacts.append(
            {
                "name": generator_output["env_impacts"].gwp.name,
                "value": generator_output["env_impacts"].gwp.value
                + (
                    retriever_output["env_impacts"].gwp.value
                    if "env_impacts" in retriever_output.keys()
                    else 0
                ),
                "unit": generator_output["env_impacts"].gwp.unit,
            }
        )
        env_impacts.append(
            {
                "name": generator_output["env_impacts"].adpe.name,
                "value": generator_output["env_impacts"].adpe.value
                + (
                    retriever_output["env_impacts"].adpe.value
                    if "env_impacts" in retriever_output.keys()
                    else 0
                ),
                "unit": generator_output["env_impacts"].adpe.unit,
            }
        )
        env_impacts.append(
            {
                "name": generator_output["env_impacts"].pe.name,
                "value": generator_output["env_impacts"].pe.value
                + (
                    retriever_output["env_impacts"].pe.value
                    if "env_impacts" in retriever_output.keys()
                    else 0
                ),
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
            retriever_output["question_entities"],
            retriever_output["queries"],
            retriever_output["llm_query"],
            retriever_output["context"],
            generator_output["answer"],
            self._merge_env_impact(retriever_output, generator_output),
        )
        return formatted_output
