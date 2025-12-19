"""
Multi-Model Summarizer Service
==============================
Routes text to the appropriate summarization model based on genre.
Supports multiple fine-tuned AraBART models for different content types.
"""

import torch
import re
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, Dict, Any

from config import settings
from nlp.extractive import ArabicExtractiveSummarizer
from nlp.genre_classifier import genre_classifier, GenreType
from nlp.keywords import extract_keywords_tfidf


class MultiModelSummarizerService:
    """
    Multi-model summarizer that routes to genre-specific models.
    
    Architecture:
    Input → Genre Classifier → Route to Model → Summarize
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
               News Model    Educational Model   Technical Model
               (XLSum)        (SummARai)         (Academic)
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractive_model = None
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self._loaded = False
        
        # Model paths configuration
        # Educational model trained on SummARai dataset
        self.model_paths = {
            "news": str(settings.ARABART_MODEL_PATH),  # Current XLSum model
            "educational": str(Path(__file__).parent.parent / "models" / "arabart_educational_final"),
            "technical": None,    # Future: academic abstracts model
            "general": str(settings.ARABART_MODEL_PATH),  # Fallback to news
        }
    
    def register_model(self, genre: GenreType, model_path: str):
        """
        Register a new model for a specific genre.
        Call this after training a new genre-specific model.
        """
        if os.path.exists(model_path):
            self.model_paths[genre] = model_path
            print(f"✅ Registered {genre} model: {model_path}")
            
            # Load the model if service is already loaded
            if self._loaded:
                self._load_single_model(genre, model_path)
        else:
            print(f"⚠️ Model path not found: {model_path}")
    
    def _load_single_model(self, genre: str, model_path: str):
        """Load a single model for a genre."""
        print(f"📥 Loading {genre} model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        model.to(self.device)
        model.eval()
        
        self.tokenizers[genre] = tokenizer
        self.models[genre] = model
        print(f"✅ {genre.capitalize()} model loaded")
    
    def load_models(self):
        """Load all available models."""
        if self._loaded:
            return
        
        print(f"🚀 Loading multi-model summarizer on {self.device.upper()}...")
        
        # Load extractive model (shared across all genres)
        print("📥 Loading extractive model (AraBERT)...")
        self.extractive_model = ArabicExtractiveSummarizer()
        
        # Load available abstractive models
        for genre, path in self.model_paths.items():
            if path and os.path.exists(path):
                self._load_single_model(genre, path)
        
        # Ensure at least one model is loaded
        if not self.models:
            raise RuntimeError("No models could be loaded!")
        
        self._loaded = True
        print(f"✅ Multi-model service ready! Available genres: {list(self.models.keys())}")
    
    def _get_model_for_genre(self, genre: GenreType):
        """Get the appropriate model for a genre, with fallback."""
        if genre in self.models:
            return self.models[genre], self.tokenizers[genre], genre
        
        # Fallback chain: educational → news → general → any available
        fallbacks = ["general", "news", "educational"]
        for fallback in fallbacks:
            if fallback in self.models:
                print(f"⚠️ No {genre} model, falling back to {fallback}")
                return self.models[fallback], self.tokenizers[fallback], fallback
        
        # Last resort: use any available model
        available = next(iter(self.models.keys()))
        return self.models[available], self.tokenizers[available], available
    
    def summarize(
        self,
        text: str,
        method: str = "textrank",
        extract_top_n: int = 5,
        max_length: int = None,
        min_length: int = None,
        force_genre: GenreType = None
    ) -> dict:
        """
        Generate summary using genre-appropriate model.
        
        Args:
            text: Input text to summarize
            method: Extraction method ('textrank' or 'mmr')
            extract_top_n: Number of sentences to extract
            max_length: Max summary length (auto-calculated if None)
            min_length: Min summary length (auto-calculated if None)
            force_genre: Override automatic genre detection
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Step 0: Preprocessing
        text = re.sub(r'(\d+)\s+\.(\d+)', r'\1.\2', text)
        
        # Step 1: Genre Classification (or use forced genre)
        if force_genre:
            genre_result = {"genre": force_genre, "confidence": 1.0, "matched_keywords": []}
        else:
            genre_result = genre_classifier.classify(text)
        
        detected_genre = genre_result["genre"]
        print(f"🏷️ Genre detected: {detected_genre} (confidence: {genre_result['confidence']})")
        
        # Step 2: Get appropriate model
        model, tokenizer, used_genre = self._get_model_for_genre(detected_genre)
        
        # Step 3: Extract keywords (returns list of (keyword, score) tuples)
        keywords = extract_keywords_tfidf(text, top_n=10)
        
        # Step 4: Extractive summarization
        extracted_sentences = self.extractive_model.summarize(
            text,
            method=method,
            top_n=extract_top_n
        )
        
        if not extracted_sentences:
            return {
                "extracted_sentences": [],
                "final_summary": "⚠️ Le texte est trop court pour être résumé.",
                "method_used": method,
                "num_sentences_extracted": 0,
                "genre": detected_genre,
                "model_used": used_genre,
                "keywords": keywords
            }
        
        # Step 5: Construct input with keywords
        sentences_text = "\n".join(extracted_sentences)
        if keywords:
            # keywords is list of (word, score) tuples, extract just the words
            keyword_words = [kw for kw, _ in keywords[:5]]
            keyword_prefix = f"الكلمات المفتاحية: {', '.join(keyword_words)} [SEP] "
            concatenated_input = keyword_prefix + sentences_text
        else:
            concatenated_input = sentences_text
        
        # Step 6: Calculate dynamic lengths based on genre
        extracted_tokens = tokenizer(concatenated_input, return_tensors="pt")["input_ids"].shape[1]
        
        # Genre-specific length tuning (used as DEFAULTS when user doesn't specify)
        if detected_genre == "educational":
            # Educational: longer, more detailed summaries
            length_ratio = 0.50
            min_ratio = 0.30
            default_min = 128  # Default minimum for educational
        elif detected_genre == "technical":
            # Technical: medium length, precise
            length_ratio = 0.40
            min_ratio = 0.20
            default_min = 100
        else:
            # News: shorter, punchy
            length_ratio = 0.35
            min_ratio = 0.15
            default_min = 64
        
        # Calculate dynamic max_length based on input size
        dynamic_max = max(default_min, min(512, int(extracted_tokens * length_ratio)))
        
        # If user provided max_length, RESPECT their choice (with minimum of 50)
        if max_length is not None:
            max_length = max(50, max_length)  # Minimum 50 tokens to avoid truncation
        else:
            # No user input: use dynamic calculation
            max_length = dynamic_max
        
        if min_length is None:
            min_length = max(20, min(max_length - 20, int(extracted_tokens * min_ratio)))
        
        # Step 7: Generate summary
        inputs = tokenizer(
            concatenated_input,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="longest"
        ).to(self.device)
        
        # Genre-specific generation parameters for factual accuracy
        if detected_genre == "educational":
            # Educational: more conservative, better factual accuracy
            gen_params = {
                "num_beams": 5,  # More beams for better quality
                "length_penalty": 0.8,  # Slightly shorter, more focused
                "repetition_penalty": 1.2,  # Avoid repetition
                "no_repeat_ngram_size": 3,  # Larger n-gram blocking
            }
        elif detected_genre == "technical":
            gen_params = {
                "num_beams": 5,
                "length_penalty": 0.9,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
            }
        else:
            # News: standard parameters
            gen_params = {
                "num_beams": 4,
                "length_penalty": 1.0,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 2,
            }
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=gen_params["num_beams"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=gen_params["length_penalty"],
                early_stopping=True,
                repetition_penalty=gen_params["repetition_penalty"],
                no_repeat_ngram_size=gen_params["no_repeat_ngram_size"]
            )
        
        raw_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Post-process: Add structure with paragraph breaks
        final_summary = self._structure_summary(raw_summary, detected_genre)
        
        return {
            "extracted_sentences": extracted_sentences,
            "final_summary": final_summary,
            "method_used": method,
            "num_sentences_extracted": len(extracted_sentences),
            "input_tokens": extracted_tokens,
            "summary_max_length": max_length,
            "keywords": keywords,
            "genre": detected_genre,
            "genre_confidence": genre_result["confidence"],
            "model_used": used_genre,
            "genre_keywords": genre_result.get("matched_keywords", [])
        }
    
    def _structure_summary(self, summary: str, genre: str) -> str:
        """
        Add structure to summary with paragraph breaks at logical topic transitions.
        Uses dynamic pattern detection instead of hardcoded terms.
        """
        if not summary or len(summary) < 100:
            return summary
        
        structured = summary
        
        # 1. DYNAMIC: Detect list items pattern "اسم: وصف" (Name: Description)
        # Only add break if the list item comes AFTER a period (sentence end)
        list_pattern = re.compile(r'(\.\s+)([ا-ي]+)(:\s)')  # Period + space + Arabic word + colon
        structured = list_pattern.sub(r'.\n\n\2\3', structured)
        
        # 2. GENERAL transition markers (linguistic, not content-specific)
        general_markers = [
            # Topic transitions
            "وتنتشر", "ومن أشهر", "كما أن", "بالإضافة إلى", "علاوة على",
            "من ناحية أخرى", "في المقابل", "أما بالنسبة", "أما",
            # Temporal/causal transitions
            "تغير هذا", "بعد ذلك", "ثم إن", "وقد", "ولقد", "نتيجة لذلك",
            # Contrast/addition
            "ومع ذلك", "لكن", "غير أن", "إلا أن",
            # Conclusion
            "وبذلك", "وهكذا", "في النهاية", "ختاماً",
        ]
        
        for marker in general_markers:
            if marker in structured and not structured.strip().startswith(marker):
                structured = structured.replace(f" {marker}", f"\n\n{marker}")
                structured = structured.replace(f". {marker}", f".\n\n{marker}")
        
        # 3. SMART: Break after long sentences (>150 chars) at period
        lines = structured.split('\n\n')
        new_lines = []
        for line in lines:
            if len(line) > 300:  # Very long paragraph
                # Split at periods, group every 2-3 sentences
                sentences = line.split('. ')
                current_para = []
                char_count = 0
                for sent in sentences:
                    current_para.append(sent)
                    char_count += len(sent)
                    if char_count > 150:  # Start new paragraph
                        new_lines.append('. '.join(current_para) + ('.' if not current_para[-1].endswith('.') else ''))
                        current_para = []
                        char_count = 0
                if current_para:
                    new_lines.append('. '.join(current_para))
            else:
                new_lines.append(line)
        
        structured = '\n\n'.join(new_lines)
        
        # Clean up
        while "\n\n\n" in structured:
            structured = structured.replace("\n\n\n", "\n\n")
        
        return structured.strip()
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def available_genres(self) -> list:
        return list(self.models.keys())


# Singleton instance
multi_model_service = MultiModelSummarizerService()
