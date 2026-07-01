from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .agent import process_chat
from .models import ChatRequest, ChatResponse
from .retriever import build_index

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    messages = [m.model_dump() for m in request.messages]

    try:
        return process_chat(messages)
    except Exception as e:
        err = str(e)
        if "429" in err or "rate limit" in err.lower():
            raise HTTPException(status_code=503, detail="LLM rate limit reached — try again shortly.")
        raise HTTPException(status_code=500, detail=f"Internal error: {err[:200]}")
