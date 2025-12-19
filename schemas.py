"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class SummarizeRequest(BaseModel):
    """Request body for summarization endpoint"""
    text: str = Field(..., min_length=50, description="Arabic text to summarize")
    method: Literal["textrank", "mmr"] = Field(default="mmr", description="Extraction method")
    extract_top_n: int = Field(default=5, ge=1, le=10, description="Number of sentences to extract")
    max_length: int = Field(default=128, ge=32, le=256, description="Max length of abstract summary")
    
class SummarizeUrlRequest(BaseModel):
    """Request body for URL summarization endpoint"""
    url: str = Field(..., description="URL of the Arabic Wikipedia article")
    method: Literal["textrank", "mmr"] = Field(default="mmr", description="Extraction method")
    extract_top_n: int = Field(default=5, ge=1, le=10, description="Number of sentences to extract")
    max_length: int = Field(default=128, ge=32, le=256, description="Max length of abstract summary")
    

class SummarizeResponse(BaseModel):
    """Response body for summarization endpoint"""
    extracted_sentences: List[str]
    final_summary: str
    method_used: str
    num_sentences_extracted: int
    input_tokens: Optional[int] = None
    summary_max_length: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
