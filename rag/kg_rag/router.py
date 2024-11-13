from dependency_injector.wiring import Provide, inject

from containers import ApplicationContainer
from fastapi import APIRouter, Body, Depends, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from kg_rag.service import KGRAG


kg_rag_router = APIRouter(prefix="/api", tags=["KG-RAG"])


@kg_rag_router.post("/kg-rag/invoke")
@inject
async def invoke(
    utterance: str = Body(title="User utterance", embed=True),
    kg_rag_service: KGRAG = Depends(Provide[ApplicationContainer.kg_rag_service]),
):
    """Invoke KG RAG with hybrid retriever and LLM generator"""
    output = kg_rag_service.run(utterance)

    return JSONResponse(
        status_code=status.HTTP_200_OK, content=jsonable_encoder(output)
    )
