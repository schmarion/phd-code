import os
import uvicorn
from fastapi import FastAPI

from containers import ApplicationContainer
from kg_rag.router import kg_rag_router
from text_rag.router import text_rag_router


app = FastAPI(
    title="Wikit KG-RAG PhD demonstrator",
    version="1.0",
    description=(
        "This API server exposes REST endpoints for the KG-RAG PhD demonstrator by Wikit R&D."
    ),
)

app.containers = ApplicationContainer()

app.include_router(kg_rag_router)
app.include_router(text_rag_router)


@app.get("/health")
async def health():
    return {"status": "OK"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
