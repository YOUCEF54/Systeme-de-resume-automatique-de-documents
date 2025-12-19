"""
Keywords Extraction using TF-IDF
"""
import re
from collections import Counter
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Arabic stopwords
ARABIC_STOPWORDS = {
    'من', 'في', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
    'التي', 'الذي', 'الذين', 'اللذين', 'اللتين', 'اللواتي', 'اللائي',
    'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم', 'أنتن',
    'كان', 'كانت', 'كانوا', 'يكون', 'تكون', 'كون', 'أن', 'إن', 'لأن',
    'أو', 'و', 'ف', 'ثم', 'لكن', 'بل', 'حتى', 'إذا', 'إذ', 'لو', 'لولا',
    'ما', 'لا', 'لم', 'لن', 'قد', 'سوف', 'سـ', 'منذ', 'حيث', 'بين',
    'كل', 'بعض', 'غير', 'كلا', 'كلتا', 'أي', 'جميع', 'عند', 'لدى',
    'فوق', 'تحت', 'أمام', 'خلف', 'قبل', 'بعد', 'خلال', 'ضد', 'نحو',
    'حول', 'دون', 'سوى', 'مثل', 'كـ', 'بـ', 'لـ', 'وـ', 'فـ',
    'أكثر', 'أقل', 'أحسن', 'أفضل', 'أول', 'آخر', 'ذات', 'ذو', 'ذي',
    'به', 'بها', 'بهم', 'له', 'لها', 'لهم', 'منه', 'منها', 'منهم',
    'عليه', 'عليها', 'عليهم', 'فيه', 'فيها', 'فيهم', 'إليه', 'إليها',
    'وهو', 'وهي', 'وكان', 'وقد', 'ولا', 'ولم', 'وإن', 'فإن', 'لأنه',
    'والتي', 'والذي', 'وأن', 'ومن', 'وفي', 'وعلى', 'ومع', 'وإلى',
    'يمكن', 'كما', 'أيضا', 'أيضاً', 'كذلك', 'هناك', 'لذلك', 'وذلك',
    'الى', 'علي', 'فى', 'او', 'ان', 'انه', 'انها', 'هذة', 'ذالك'
}


def clean_arabic_text(text: str) -> str:
    """Remove diacritics and normalize Arabic text"""
    # Remove Arabic diacritics
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)
    
    # Normalize alef variants
    text = re.sub('[إأآا]', 'ا', text)
    
    # Normalize taa marbuta
    text = re.sub('ة', 'ه', text)
    
    # Remove non-Arabic characters except spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_keywords_tfidf(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract keywords using TF-IDF.
    
    Args:
        text: Arabic text
        top_n: Number of keywords to extract
        
    Returns:
        List of (keyword, score) tuples
    """
    # Clean and normalize text
    cleaned_text = clean_arabic_text(text)
    
    # Split into words
    words = cleaned_text.split()
    
    # Filter stopwords and short words
    filtered_words = [
        w for w in words 
        if w not in ARABIC_STOPWORDS and len(w) > 2
    ]
    
    if not filtered_words:
        return []
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Calculate TF-IDF-like scores
    total_words = len(filtered_words)
    unique_words = len(word_freq)
    
    # Simple TF-IDF: TF * log(N/df) - here we use frequency as proxy
    keywords = []
    for word, freq in word_freq.most_common(top_n * 2):
        tf = freq / total_words
        # Boost longer words (often more meaningful)
        length_boost = min(len(word) / 5, 1.5)
        score = tf * length_boost
        keywords.append((word, round(score, 4)))
    
    # Sort by score and return top_n
    keywords.sort(key=lambda x: x[1], reverse=True)
    return keywords[:top_n]


def extract_keywords_from_sentences(sentences: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract keywords from a list of sentences using TF-IDF vectorizer.
    """
    if not sentences:
        return []
    
    # Join all sentences
    full_text = " ".join(sentences)
    
    return extract_keywords_tfidf(full_text, top_n)
