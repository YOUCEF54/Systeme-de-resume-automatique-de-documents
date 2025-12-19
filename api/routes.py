"""
API Routes for Summarization
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from typing import Literal, Optional, List

from services.summarizer import summarizer_service
from api.schemas import SummarizeRequest, SummarizeUrlRequest, SummarizeResponse, HealthResponse, KeywordsResponse
from utils.file_parsers import extract_text_from_file, get_supported_extensions
from utils.exporters import export_to_pdf, export_to_docx
from nlp.keywords import extract_keywords_tfidf
from utils.scraper import scrape_wikipedia_article


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy",
        model_loaded=summarizer_service.is_loaded
    )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Generate a hybrid summary of Arabic text.
    """
    try:
        result = summarizer_service.summarize(
            text=request.text,
            method=request.method,
            extract_top_n=request.extract_top_n,
            max_length=request.max_length
        )
        
        # Extract keywords
        keywords = extract_keywords_tfidf(request.text, top_n=10)
        
        return SummarizeResponse(
            extracted_sentences=result["extracted_sentences"],
            final_summary=result["final_summary"],
            method_used=result["method_used"],
            num_sentences_extracted=result["num_sentences_extracted"],
            input_tokens=result.get("input_tokens"),
            summary_max_length=result.get("summary_max_length"),
            keywords=keywords
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


@router.post("/summarize-smart", response_model=SummarizeResponse)
async def summarize_smart(request: SummarizeRequest):
    """
    Smart summarization with automatic genre detection and model routing.
    
    - Detects genre: news, educational, technical
    - Routes to genre-optimized model
    - Returns genre metadata in response
    """
    try:
        # Import multi-model service
        from services.multi_model_summarizer import multi_model_service
        
        # Ensure models are loaded
        if not multi_model_service.is_loaded:
            multi_model_service.load_models()
        
        result = multi_model_service.summarize(
            text=request.text,
            method=request.method,
            extract_top_n=request.extract_top_n,
            max_length=request.max_length,
            force_genre=request.genre  # Allow genre override
        )
        
        return SummarizeResponse(
            extracted_sentences=result["extracted_sentences"],
            final_summary=result["final_summary"],
            method_used=result["method_used"],
            num_sentences_extracted=result["num_sentences_extracted"],
            input_tokens=result.get("input_tokens"),
            summary_max_length=result.get("summary_max_length"),
            keywords=result.get("keywords"),
            genre=result.get("genre"),
            genre_confidence=result.get("genre_confidence"),
            model_used=result.get("model_used")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart summarization error: {str(e)}")


@router.post("/summarize-file", response_model=SummarizeResponse)
async def summarize_file(
    file: UploadFile = File(...),
    method: Literal["textrank", "mmr"] = Form(default="mmr"),
    extract_top_n: int = Form(default=5, ge=1, le=10),
    max_length: Optional[int] = Form(default=None, ge=32, le=512)
):
    """
    Upload a file (.txt, .pdf, .docx) and generate a hybrid summary.
    """
    # Validate file type
    supported = get_supported_extensions()
    if not any(file.filename.lower().endswith(ext) for ext in supported):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(supported)}"
        )
    
    try:
        content = await file.read()
        text = extract_text_from_file(file.filename, content)
        
        if not text or len(text) < 50:
            raise HTTPException(status_code=400, detail="File content too short (min 50 characters)")
        
        result = summarizer_service.summarize(
            text=text,
            method=method,
            extract_top_n=extract_top_n,
            max_length=max_length
        )
        
        keywords = extract_keywords_tfidf(text, top_n=10)
        
        return SummarizeResponse(
            extracted_sentences=result["extracted_sentences"],
            final_summary=result["final_summary"],
            method_used=result["method_used"],
            num_sentences_extracted=result["num_sentences_extracted"],
            input_tokens=result.get("input_tokens"),
            summary_max_length=result.get("summary_max_length"),
            keywords=keywords
        )
    
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


@router.post("/summarize-url", response_model=SummarizeResponse)
async def summarize_url(request: SummarizeUrlRequest):
    """
    Scrape text from a Wikipedia URL and generate a hybrid summary.
    """
    try:
        # Scrape content
        text = scrape_wikipedia_article(request.url)
        
        if not text or len(text) < 50:
            raise HTTPException(status_code=400, detail="Could not extract sufficient text from the URL. Ensure it's a valid Wikipedia article.")
        
        result = summarizer_service.summarize(
            text=text,
            method=request.method,
            extract_top_n=request.extract_top_n,
            max_length=request.max_length
        )
        
        # Extract keywords
        keywords = extract_keywords_tfidf(text, top_n=10)
        
        return SummarizeResponse(
            extracted_sentences=result["extracted_sentences"],
            final_summary=result["final_summary"],
            method_used=result["method_used"],
            num_sentences_extracted=result["num_sentences_extracted"],
            input_tokens=result.get("input_tokens"),
            summary_max_length=result.get("summary_max_length"),
            keywords=keywords
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL summarization error: {str(e)}")


@router.post("/keywords", response_model=KeywordsResponse)
async def extract_keywords(request: SummarizeRequest):
    """
    Extract keywords from Arabic text.
    """
    try:
        keywords = extract_keywords_tfidf(request.text, top_n=15)
        return KeywordsResponse(keywords=keywords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keyword extraction error: {str(e)}")


@router.post("/export/pdf")
async def export_pdf(request: SummarizeRequest):
    """
    Generate summary and export to PDF.
    """
    try:
        result = summarizer_service.summarize(
            text=request.text,
            method=request.method,
            extract_top_n=request.extract_top_n,
            max_length=request.max_length
        )
        
        keywords = extract_keywords_tfidf(request.text, top_n=10)
        
        pdf_bytes = export_to_pdf(
            original_text=request.text,
            extracted_sentences=result["extracted_sentences"],
            final_summary=result["final_summary"],
            keywords=keywords,
            method_used=result["method_used"]
        )
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=summary.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export error: {str(e)}")


@router.post("/export/docx")
async def export_docx(request: SummarizeRequest):
    """
    Generate summary and export to Word document.
    """
    try:
        result = summarizer_service.summarize(
            text=request.text,
            method=request.method,
            extract_top_n=request.extract_top_n,
            max_length=request.max_length
        )
        
        keywords = extract_keywords_tfidf(request.text, top_n=10)
        
        docx_bytes = export_to_docx(
            original_text=request.text,
            extracted_sentences=result["extracted_sentences"],
            final_summary=result["final_summary"],
            keywords=keywords,
            method_used=result["method_used"]
        )
        
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=summary.docx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word export error: {str(e)}")
