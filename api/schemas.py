"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple


class SummarizeRequest(BaseModel):
    """Request body for summarization endpoint"""
    text: str = Field(..., min_length=50, description="Arabic text to summarize")
    method: Literal["textrank", "mmr"] = Field(default="mmr", description="Extraction method")
    extract_top_n: int = Field(default=5, ge=1, le=10, description="Number of sentences to extract")
    max_length: Optional[int] = Field(default=None, ge=32, le=512, description="Max length of summary (None=auto)")
    genre: Optional[Literal["news", "educational", "technical", "general"]] = Field(
        default=None, 
        description="Force genre (None=auto-detect)"
    )


class SummarizeUrlRequest(BaseModel):
    """Request body for URL summarization endpoint"""
    url: str = Field(..., description="URL of the Arabic Wikipedia article")
    method: Literal["textrank", "mmr"] = Field(default="mmr", description="Extraction method")
    extract_top_n: int = Field(default=5, ge=1, le=10, description="Number of sentences to extract")
    max_length: Optional[int] = Field(default=None, ge=32, le=512, description="Max length of summary (None=auto)")
    genre: Optional[Literal["news", "educational", "technical", "general"]] = Field(default=None)


class SummarizeResponse(BaseModel):
    """Response body for summarization endpoint"""
    extracted_sentences: List[str]
    final_summary: str
    method_used: str
    num_sentences_extracted: int
    input_tokens: Optional[int] = None
    summary_max_length: Optional[int] = None
    keywords: Optional[List[Tuple[str, float]]] = None
    # Genre detection fields (multi-model)
    genre: Optional[str] = None
    genre_confidence: Optional[float] = None
    model_used: Optional[str] = None


class KeywordsResponse(BaseModel):
    """Response for keywords endpoint"""
    keywords: List[Tuple[str, float]]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
