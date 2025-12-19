"""
Hybrid Summarizer Service - Combines Extractive + Abstractive
"""
import torch
import re  # Added for text preprocessing
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import settings
from nlp.extractive import ArabicExtractiveSummarizer


# =============================================================================
# KEYWORD EXTRACTION (Domain-Aware Enhancement)
# =============================================================================

from nlp.keywords import extract_keywords_tfidf as extract_keywords


class HybridSummarizerService:
    """
    Service class for hybrid Arabic summarization.
    Loads models once at startup and reuses them for all requests.
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractive_model = None
        self.abstractive_model = None
        self.tokenizer = None
        self._loaded = False
    
    def load_models(self):
        """Load both extractive and abstractive models"""
        if self._loaded:
            return
        
        print(f"🚀 Loading models on {self.device.upper()}...")
        
        # Load extractive model (AraBERT for embeddings)
        print("📥 Loading extractive model...")
        self.extractive_model = ArabicExtractiveSummarizer()
        
        # Load abstractive model (fine-tuned AraBART)
        model_path = str(settings.ARABART_MODEL_PATH)
        print(f"📥 Loading abstractive model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        self.abstractive_model.to(self.device)
        self.abstractive_model.eval()
        
        self._loaded = True
        print("✅ All models loaded successfully!")
    
    def summarize(
        self,
        text: str,
        method: str = "mmr",
        extract_top_n: int = 5,
        max_length: int = None,
        min_length: int = None
    ) -> dict:
        """
        Generate hybrid summary with keyword-guided input.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Step 0: Preprocessing (Fix number formatting spaces like "88 .5%" -> "88.5%")
        # This fixes tokenization issues where numbers are split
        text = re.sub(r'(\d+)\s+\.(\d+)', r'\1.\2', text)
        
        # Step 0.5: Extract keywords from original text (BEFORE extraction)
        keywords = extract_keywords(text, top_n=5)
        
        # Step 1: Extractive summarization
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
                "keywords": keywords
            }
        
        # Step 2: Check minimum sentences for abstractive phase
        if len(extracted_sentences) < settings.MIN_SENTENCES_FOR_ABSTRACTIVE:
            return {
                "extracted_sentences": extracted_sentences,
                "final_summary": " ".join(extracted_sentences) + "\n\n⚠️ Texte trop court pour la phase abstractive.",
                "method_used": method,
                "num_sentences_extracted": len(extracted_sentences),
                "input_tokens": 0,
                "summary_max_length": 0,
                "keywords": keywords
            }
        
        # Step 3: Construct keyword-guided input for AraBART
        # Format: "الكلمات المفتاحية: [keywords] [SEP] [sentences]"
        sentences_text = "\n".join(extracted_sentences)
        
        if keywords:
            # Use a simpler prompt format that the model might understand better, or just prepend without SEP if model is standard mBART/AraBART
            # Trying a more natural prompt if it's fine-tuned on text. 
            # If the model is just a standard summarizer, adding "Keywords: ..." might confuse it.
            # Let's try prepending keywords with a simple separator.
            keyword_text = " ".join(keywords)
            concatenated_input = f"{keyword_text} . {sentences_text}"
        else:
            concatenated_input = sentences_text
            
        print(f"DEBUG INPUT TO ARABART:\n{concatenated_input}\n---")

        
        # Step 4: Calculate dynamic length
        extracted_tokens = self.tokenizer(concatenated_input, return_tensors="pt")["input_ids"].shape[1]
        
        # Check if user provided max_length
        user_defined_max_length = max_length is not None

        if max_length is None:
            # OPTIMIZED: Ratio set to 0.33 to allow enough space for multi-topic summaries (Macron case)
            max_length = max(40, min(512, int(extracted_tokens * 0.33)))
        
        if min_length is None:
            if user_defined_max_length:
                # OPTIMIZED: Increased floor to 40% to respect user's "Long" setting.
                # Input-dependent cap (80% of input) prevents hallucinations if input is short.
                target_min = int(max_length * 0.40)
                min_length = min(target_min, int(extracted_tokens * 0.8))
                min_length = max(20, min_length) # Absolute floor of 20 tokens
            else:
                # OPTIMIZED: Increased floor to 0.20 to FORCE detail and prevent single-sentence summaries
                # Old: ... * 0.10 -> New: ... * 0.20
                min_length = max(16, min(max_length - 10, int(extracted_tokens * 0.20)))
        
        # Step 5: Abstractive summarization
        inputs = self.tokenizer(
            concatenated_input,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="longest"
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.abstractive_model.generate(
                inputs["input_ids"],
                num_beams=4,  # Beam search for logical coherence
                max_length=max_length,
                min_length=min_length,  # Dynamic
                length_penalty=1.0,  # Neutral
                early_stopping=True,
                repetition_penalty=1.2,  # Slight penalty to reduce loops, but not kill flow
                no_repeat_ngram_size=3   # Relaxed from 2 to 3 to allow common bigrams (e.g. "في ال")
            )
        
        final_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return {
            "extracted_sentences": extracted_sentences,
            "final_summary": final_summary,
            "method_used": method,
            "num_sentences_extracted": len(extracted_sentences),
            "input_tokens": extracted_tokens,
            "summary_max_length": max_length,
            "keywords": keywords  # Domain-aware keywords
        }
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton instance
summarizer_service = HybridSummarizerService()
