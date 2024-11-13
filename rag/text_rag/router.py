from dependency_injector.wiring import Provide, inject

from containers import ApplicationContainer
from fastapi import APIRouter, Body, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from text_rag.service import TextRAG


text_rag_router = APIRouter(prefix="/api", tags=["Text-RAG"])


@text_rag_router.post("/text-rag/invoke")
@inject
async def invoke(
    utterance: str = Body(title="User utterance", embed=True),
    text_rag_service: TextRAG = Depends(Provide[ApplicationContainer.text_rag_service]),
):
    """Invoke Text RAG with retriever based on embeddings similarity and LLM generator."""
    output = text_rag_service.run(utterance)

    return JSONResponse(
        status_code=status.HTTP_200_OK, content=jsonable_encoder(output)
    )
