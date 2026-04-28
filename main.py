"""Prompt Wars — Enhancement Engine API.

A prompt enhancement service that transforms raw prompts into polished,
context-aware instructions using configurable enhancement strategies.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prompt_wars")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Strategy(str, Enum):
    """Available prompt-enhancement strategies."""

    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    PERSUASIVE = "persuasive"


class PromptRequest(BaseModel):
    """Incoming prompt payload."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Raw prompt text to enhance (3–5 000 chars).",
        json_schema_extra={"examples": ["Tell me about AI"]},
    )
    strategy: Strategy = Field(
        default=Strategy.PROFESSIONAL,
        description="Enhancement strategy to apply.",
    )
    tone: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Optional tone override, e.g. 'formal', 'casual', 'empathetic'.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt text must not be blank or whitespace-only.")
        return v.strip()


class EnhancedPromptResponse(BaseModel):
    """Response returned after enhancement."""

    id: str
    original: str
    enhanced: str
    strategy: str
    tone: Optional[str] = None
    word_count_original: int
    word_count_enhanced: int
    enhancement_ratio: float
    timestamp: str
    version: str = API_VERSION


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str
    timestamp: str


class StatsResponse(BaseModel):
    total_enhancements: int
    by_strategy: dict[str, int]


# ---------------------------------------------------------------------------
# Enhancement Engine
# ---------------------------------------------------------------------------

_STRATEGY_TEMPLATES: dict[Strategy, str] = {
    Strategy.PROFESSIONAL: (
        "You are an expert assistant. {tone_clause}"
        "Provide a comprehensive, well-structured, and detailed response to the "
        "following request. Use clear headings, bullet points where appropriate, "
        "and ensure factual accuracy:\n\n\"{text}\""
    ),
    Strategy.CREATIVE: (
        "You are a world-class creative writer. {tone_clause}"
        "Craft a compelling, vivid, and emotionally resonant response. "
        "Use sensory language, metaphor, and narrative structure where "
        "appropriate:\n\n\"{text}\""
    ),
    Strategy.TECHNICAL: (
        "You are a senior software engineer and technical architect. {tone_clause}"
        "Provide a thorough, technically accurate response with code examples "
        "where relevant, best-practice recommendations, and edge-case "
        "considerations:\n\n\"{text}\""
    ),
    Strategy.ACADEMIC: (
        "You are a distinguished academic researcher. {tone_clause}"
        "Provide a rigorous, evidence-based response. Structure your answer "
        "with an introduction, analysis, and conclusion. Cite sources where "
        "applicable:\n\n\"{text}\""
    ),
    Strategy.PERSUASIVE: (
        "You are a master communicator and persuasion expert. {tone_clause}"
        "Craft a powerful, convincing response using rhetorical techniques, "
        "strong evidence, and a clear call to action:\n\n\"{text}\""
    ),
}

STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "professional": "Clear, polished, and business-ready prompts.",
    "creative": "Vivid, imaginative, and narratively rich prompts.",
    "technical": "Precise, structured, and engineering-focused prompts.",
    "academic": "Formal, evidence-based, and scholarly prompts.",
    "persuasive": "Compelling, action-oriented, and rhetorically strong prompts.",
}


def enhance_prompt(text: str, strategy: Strategy, tone: Optional[str] = None) -> str:
    """Apply the selected enhancement strategy to raw prompt text."""
    tone_clause = f'Adopt a "{tone}" tone throughout. ' if tone else ""
    template = _STRATEGY_TEMPLATES[strategy]
    return template.format(text=text, tone_clause=tone_clause)


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------


class AppState:
    """Mutable application state — avoids loose globals."""

    def __init__(self) -> None:
        self.startup_time: datetime = datetime.now(timezone.utc)
        self.enhancement_count: int = 0
        self.strategy_counts: dict[str, int] = {s.value: 0 for s in Strategy}

    def record(self, strategy: Strategy) -> None:
        self.enhancement_count += 1
        self.strategy_counts[strategy.value] += 1

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.startup_time).total_seconds()


state = AppState()

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Prompt Wars Enhancement Engine v%s starting up", API_VERSION)
    logger.info("Loaded %d enhancement strategies", len(Strategy))
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prompt Wars — Enhancement Engine",
    description=(
        "Transforms raw prompts into polished, context-aware instructions "
        "using configurable enhancement strategies."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["General"], summary="Root")
def root():
    """Confirms the API is live and lists available endpoints."""
    return {
        "service": "Prompt Wars Enhancement Engine",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": {
            "POST /enhance": "Enhance a prompt",
            "GET /strategies": "List available strategies",
            "GET /health": "Health check",
            "GET /stats": "Usage statistics",
        },
    }


@app.get("/health", tags=["General"], summary="Health Check", response_model=HealthResponse)
def health():
    """Returns service health and uptime."""
    now = datetime.now(timezone.utc)
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(state.uptime_seconds, 2),
        version=API_VERSION,
        timestamp=now.isoformat(),
    )


@app.get("/strategies", tags=["Enhancement"], summary="List Strategies")
def strategies():
    """Returns all available enhancement strategies with descriptions."""
    return [
        {"name": s.value, "description": STRATEGY_DESCRIPTIONS[s.value]}
        for s in Strategy
    ]


@app.post(
    "/enhance",
    tags=["Enhancement"],
    summary="Enhance a Prompt",
    response_model=EnhancedPromptResponse,
)
def enhance(prompt: PromptRequest):
    """
    Enhance a raw prompt using the selected strategy.

    - **text** — the raw prompt (3–5 000 characters)
    - **strategy** — one of: professional, creative, technical, academic, persuasive
    - **tone** — optional tone override (e.g. formal, casual)
    """
    enhanced_text = enhance_prompt(prompt.text, prompt.strategy, prompt.tone)
    state.record(prompt.strategy)

    orig_words = len(prompt.text.split())
    enhanced_words = len(enhanced_text.split())
    request_id = str(uuid.uuid4())

    logger.info(
        "id=%s strategy=%s words=%d->%d",
        request_id[:8],
        prompt.strategy.value,
        orig_words,
        enhanced_words,
    )

    return EnhancedPromptResponse(
        id=request_id,
        original=prompt.text,
        enhanced=enhanced_text,
        strategy=prompt.strategy.value,
        tone=prompt.tone,
        word_count_original=orig_words,
        word_count_enhanced=enhanced_words,
        enhancement_ratio=round(enhanced_words / max(orig_words, 1), 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/stats", tags=["General"], summary="Usage Stats", response_model=StatsResponse)
def stats():
    """Returns usage statistics for the enhancement engine."""
    return StatsResponse(
        total_enhancements=state.enhancement_count,
        by_strategy=state.strategy_counts,
    )


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
        },
    )