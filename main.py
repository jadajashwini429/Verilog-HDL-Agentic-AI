import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import run_verilog_agent

app = FastAPI(title="Verilog Agentic AI", version="2.0.0")

# Kept open for simple development; production frontend is served by this same API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class GenerateRequest(BaseModel):
    user_request: str = Field(min_length=3, max_length=10000)


@app.get("/health")
def health():
    return {"status": "ok", "service": "verilog-agentic-ai", "version": "2.0.0"}


@app.post("/generate")
def generate(body: GenerateRequest):
    try:
        return run_verilog_agent(body.user_request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
