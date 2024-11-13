import json
import os

from litellm import embedding
from sklearn.metrics.pairwise import cosine_similarity


class TextSimilarityRetriever:
    """Text retriever based on embeddings similarity for the RAG task."""

    def __init__(self) -> None:
        self.embedding_model = os.environ.get("EMBEDDING_MODEL")
        self.chunks = self._load_chunks()
        self.chunks_embeddings = self._embed_chunks()

    def _load_chunks(self) -> list[dict[str, any]]:
        """Load chunks from a given file."""
        with open("./retriever/chunks.json", encoding="utf-8") as json_data:
            chunks = json.load(json_data)
            json_data.close()
        return chunks

    def _embed_chunks(self) -> list[list[float]]:
        """Create embeddings for all chunks."""
        chunks_embeddings = []
        for chunk in self.chunks:
            embed_chunk = embedding(
                model=self.embedding_model, input=[chunk["text"]]
            ).data[0]["embedding"]
            chunk[embedding] = embed_chunk
            chunks_embeddings.append(embed_chunk)
        return chunks_embeddings

    def _find_relevant_chunks(self, embed_question) -> list[str]:
        """Extract the 3 most relevant chunks for a given question based on cosine similarity."""
        chunks = []
        sim = cosine_similarity([embed_question], self.chunks_embeddings)[0]
        for index in sim.argsort()[-3:][::-1]:
            chunks.append(self.chunks[index]["text"])
        return chunks

    def _format_context(self, chunks: list[str]) -> str:
        """Format context as a string."""
        if len(chunks) == 0:
            context = "[]"
        else:
            context = "\n\n-----\n\n".join(chunks)
        return context

    def run(self, question: str) -> list[str]:
        """Retrieve context for a question based on a chunks."""
        embed_question = embedding(model=self.embedding_model, input=[question]).data[
            0
        ]["embedding"]
        relevant_chunks = self._find_relevant_chunks(embed_question)
        output = {"context": self._format_context(relevant_chunks)}
        return output
