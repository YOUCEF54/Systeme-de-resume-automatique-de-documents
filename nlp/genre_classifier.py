"""
Genre Classifier for Arabic Text
================================
Classifies input text into genres (news, educational, technical)
to route to the appropriate summarization model.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from typing import Literal

GenreType = Literal["news", "educational", "technical", "general"]


class ArabicGenreClassifier:
    """
    Lightweight genre classifier for Arabic text.
    Uses keyword-based heuristics + optional ML model.
    """
    
    # Genre-specific keyword patterns
    NEWS_KEYWORDS = [
        "قال", "أعلن", "صرح", "أكد", "كشف",  # Said, announced, stated
        "اليوم", "أمس", "غداً", "الأسبوع",  # Today, yesterday, tomorrow
        "وكالة", "مراسل", "صحيفة", "قناة",  # Agency, reporter, newspaper
        "الرئيس", "الوزير", "الحكومة", "البرلمان",  # President, minister
        "حادث", "اعتقال", "تظاهر", "انفجار",  # Incident, arrest, protest
        "دولار", "سعر", "بورصة", "أسهم",  # Dollar, price, stock
        "مباراة", "فريق", "لاعب", "بطولة",  # Match, team, player
    ]
    
    EDUCATIONAL_KEYWORDS = [
        # Definitions & Concepts
        "تعني", "يُعرّف", "المفهوم", "النظرية", "مفهوم",
        "التعريف", "المصطلح", "مصطلح", "الأصل",
        # Academic markers
        "تاريخياً", "علمياً", "فلسفياً", "ثقافياً", "اجتماعياً", "سياسياً", "نفسياً",
        "الدراسات", "البحث", "الأبحاث", "علم النفس", "علم الاجتماع",
        # Classification & Structure
        "يُصنّف", "ينقسم", "أنواع", "يختلف جذرياً",
        "مثال", "على سبيل المثال", "بمعنى آخر",
        "وفقاً لـ", "حسب", "بناءً على",
        # Literary & Books
        "الكتاب", "المؤلف", "الرواية", "الأدب", "الأدبية",
        # Sciences
        "العلوم", "الرياضيات", "الفيزياء",
        # Analysis & Essay markers
        "تأثير", "تأثيره", "إشكاليات", "الإشكاليات",
        "جوهرياً", "أساساً", "بالدرجة الأولى",
        "من ناحية", "من جهة أخرى", "في المقابل",
        "يرى", "يعتقد", "يشير إلى",
        # Generational/Social analysis terms (common in essays)
        "الجيل", "جيل", "الأجيال", "المجتمع", "المجتمعات",
        "الهوية", "الانتماء", "القيم", "السلوك",
        "التحول", "التغير", "تبدل", "تطور",
    ]
    
    TECHNICAL_KEYWORDS = [
        # Programming & Software
        "خوارزمية", "برمجة", "كود", "نظام",
        "قاعدة بيانات", "سيرفر", "شبكة",
        "تقنية", "تكنولوجيا", "رقمي", "الرقمية",
        "ذكاء اصطناعي", "تعلم آلي",
        "API", "HTTP", "JSON", "Python",
        # Hardware
        "معالج", "ذاكرة", "تخزين",
        # Digital/Internet terms (technical contexts)
        "الإنترنت", "المنصات", "التطبيقات", "البرامج",
        "الهواتف الذكية", "الأجهزة",
    ]
    
    def __init__(self):
        """Initialize the classifier."""
        self.vectorizer = None
        self.model = None
        self._trained = False
    
    def classify_by_keywords(self, text: str) -> tuple[GenreType, float]:
        """
        Fast keyword-based classification.
        Returns (genre, confidence_score).
        """
        text_lower = text.lower()
        
        # Count keyword matches for each genre
        news_score = sum(1 for kw in self.NEWS_KEYWORDS if kw in text)
        edu_score = sum(1 for kw in self.EDUCATIONAL_KEYWORDS if kw in text)
        tech_score = sum(1 for kw in self.TECHNICAL_KEYWORDS if kw in text)
        
        total = news_score + edu_score + tech_score
        
        if total == 0:
            return "general", 0.5
        
        # Calculate confidence
        scores = {"news": news_score, "educational": edu_score, "technical": tech_score}
        best_genre = max(scores, key=scores.get)
        confidence = scores[best_genre] / total
        
        # Require minimum confidence
        if confidence < 0.4:
            return "general", confidence
        
        return best_genre, confidence
    
    def classify(self, text: str) -> dict:
        """
        Main classification method.
        Returns genre classification with metadata.
        """
        genre, confidence = self.classify_by_keywords(text)
        
        # Get keyword matches for explanation
        matches = {
            "news": [kw for kw in self.NEWS_KEYWORDS if kw in text][:3],
            "educational": [kw for kw in self.EDUCATIONAL_KEYWORDS if kw in text][:3],
            "technical": [kw for kw in self.TECHNICAL_KEYWORDS if kw in text][:3],
        }
        
        return {
            "genre": genre,
            "confidence": round(confidence, 2),
            "matched_keywords": matches.get(genre, []),
            "all_scores": {
                "news": len([kw for kw in self.NEWS_KEYWORDS if kw in text]),
                "educational": len([kw for kw in self.EDUCATIONAL_KEYWORDS if kw in text]),
                "technical": len([kw for kw in self.TECHNICAL_KEYWORDS if kw in text]),
            }
        }


# Singleton instance
genre_classifier = ArabicGenreClassifier()


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    classifier = ArabicGenreClassifier()
    
    # Test news
    news_text = "أعلن الرئيس اليوم عن خطة جديدة للاقتصاد، وقال الوزير إن الحكومة ستتخذ إجراءات."
    print("News text:", classifier.classify(news_text))
    
    # Test educational
    edu_text = "تعني الديستوبيا في اللغة اليونانية المكان الخبيث. ومن أشهر الأعمال الأدبية رواية 1984."
    print("Educational text:", classifier.classify(edu_text))
    
    # Test technical
    tech_text = "تستخدم الخوارزمية قاعدة بيانات لتخزين البيانات، مع API للتواصل مع السيرفر."
    print("Technical text:", classifier.classify(tech_text))
