"""
File Parsers for PDF and Word documents
"""
import io
from typing import Optional


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract text from PDF file bytes.
    
    Args:
        content: PDF file content as bytes
        
    Returns:
        Extracted text as string
    """
    import fitz  # PyMuPDF
    
    text_parts = []
    
    # Open PDF from bytes
    pdf_document = fitz.open(stream=content, filetype="pdf")
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text_parts.append(page.get_text())
    
    pdf_document.close()
    
    return "\n\n".join(text_parts)


def extract_text_from_docx(content: bytes) -> str:
    """
    Extract text from Word document bytes.
    
    Args:
        content: DOCX file content as bytes
        
    Returns:
        Extracted text as string
    """
    from docx import Document
    
    # Open document from bytes
    doc = Document(io.BytesIO(content))
    
    paragraphs = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)
    
    return "\n\n".join(paragraphs)


def extract_text_from_file(filename: str, content: bytes) -> Optional[str]:
    """
    Extract text from file based on extension.
    
    Args:
        filename: Name of the file
        content: File content as bytes
        
    Returns:
        Extracted text or None if unsupported format
    """
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.txt'):
        return content.decode('utf-8')
    elif filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(content)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(content)
    else:
        return None


def get_supported_extensions() -> list:
    """Return list of supported file extensions"""
    return ['.txt', '.pdf', '.docx']
