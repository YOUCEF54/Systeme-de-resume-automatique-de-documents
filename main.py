"""
FastAPI Backend for Arabic Hybrid Summarization
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from api.routes import router
from services.summarizer import summarizer_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    print("🚀 Starting Arabic Summarization API...")
    summarizer_service.load_models()
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Arabic Summarization API",
    description="Hybrid extractive + abstractive summarization for Arabic text",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
