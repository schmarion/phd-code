import os

import litellm
from ecologits import EcoLogits


class LLMGenerator:
    """Generator based on LLM for the RAG task."""

    def __init__(self) -> None:
        self.model_name = os.environ.get("GENERATION_MODEL")

    def _generate_prompt(self, context: str, question: str) -> list[dict[str, str]]:
        """Generate contextualised prompt for the LLM model."""
        prompt = [
            {
                "role": "system",
                "content": """Tu es un assistant qui aide à formuler des réponses jolies et compréhensibles pour les humains.""",
            },
            {
                "role": "user",
                "content": """Utilise la partie "Information" pour répondre à la question.
            Les informations fournies font foi, tu ne dois pas les modifier ou les compléter.
            S'il n'y a pas d'information fournie, répond "Je suis désolée mais je n'ai pas la réponse à votre question.", n'utilise pas tes connaissances pour répondre.
            Ne mentionne pas que tu bases ta réponse sur les informations fournies.""",
            },
            {
                "role": "user",
                "content": f"""Information: {context}
            Question: {question}""",
            },
        ]
        return prompt

    def _generate_answer(self, prompt: list[dict[str, str]]) -> dict[str, any]:
        """Generate answer with a LLM based on a given prompt."""
        EcoLogits.init(providers="litellm", electricity_mix_zone="SWE")
        response = litellm.completion(
            model=self.model_name, messages=prompt, temperature=0.0
        )
        output = {
            "answer": response["choices"][0]["message"]["content"],
            "env_impacts": response.impacts,
        }
        return output

    def run(self, context: str, question: str) -> dict[str, any]:
        """Generate answer with a contextualised prompt for a given question."""
        prompt = self._generate_prompt(context, question)
        output = self._generate_answer(prompt)
        return output
