"""
Hybrid Summarizer Service - Combines Extractive + Abstractive
"""
import sys
import os

# Add parent directory to path to import extractive_summarizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from extractive_summarizer import ArabicExtractiveSummarizer


class HybridSummarizerService:
    """
    Service class for hybrid Arabic summarization.
    Loads models once at startup and reuses them for all requests.
    """
    
    def __init__(self, model_path: str = "models/arabart_finetuned"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
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
        print(f"📥 Loading abstractive model from {self.model_path}...")
        abs_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            self.model_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(abs_model_path)
        self.abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(abs_model_path)
        self.abstractive_model.to(self.device)
        self.abstractive_model.eval()
        
        self._loaded = True
        print("✅ All models loaded successfully!")
    
    def summarize(
        self,
        text: str,
        method: str = "mmr",
        extract_top_n: int = 5,
        max_length: int = None,  # None means auto-calculate
        min_length: int = None   # None means auto-calculate
    ) -> dict:
        """
        Generate hybrid summary.
        
        Args:
            text: Arabic text to summarize
            method: 'textrank' or 'mmr'
            extract_top_n: Number of sentences to extract
            max_length: Max length of final summary (None = auto)
            min_length: Min length of final summary (None = auto)
            
        Returns:
            dict with extracted_sentences and final_summary
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
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
                "num_sentences_extracted": 0
            }
        
        # Step 2: Check minimum sentences for abstractive phase
        # If too few sentences, skip abstractive to avoid hallucination
        if len(extracted_sentences) < 3:
            return {
                "extracted_sentences": extracted_sentences,
                "final_summary": " ".join(extracted_sentences) + "\n\n⚠️ Texte trop court pour la phase abstractive (< 3 phrases). Résumé extractif uniquement.",
                "method_used": method,
                "num_sentences_extracted": len(extracted_sentences),
                "input_tokens": 0,
                "summary_max_length": 0
            }
        
        # Step 3: Concatenate extracted sentences
        concatenated_input = " ".join(extracted_sentences)
        
        # Step 3: Calculate dynamic length based on input
        # Tokenize to get actual token count
        input_tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        extracted_tokens = self.tokenizer(concatenated_input, return_tensors="pt")["input_ids"].shape[1]
        
        # Summary should be ~25-40% of extracted text length
        # with bounds between 64 and 512 tokens
        if max_length is None:
            max_length = max(64, min(512, int(extracted_tokens * 0.5)))
        
        if min_length is None:
            min_length = max(32, min(max_length - 20, int(extracted_tokens * 0.25)))
        
        # Step 4: Abstractive summarization
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
                num_beams=4,
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.5,  # Reduced to allow longer text
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        final_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return {
            "extracted_sentences": extracted_sentences,
            "final_summary": final_summary,
            "method_used": method,
            "num_sentences_extracted": len(extracted_sentences),
            "input_tokens": input_tokens,
            "summary_max_length": max_length
        }
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
