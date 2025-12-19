"""
Partie 1 : Résumé Extractif pour Textes Arabes
================================================
Architecture hybride : TextRank + MMR
Optimisé pour CPU uniquement
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
from spacy.lang.ar import Arabic


# =============================================================================
# MODULE 1 : CONFIGURATION
# =============================================================================

class ArabicExtractiveConfig:
    """Configuration pour le résumeur extractif arabe"""
    
    # Modèle AraBERT
    MODEL_NAME = "aubmindlab/bert-base-arabertv02"
    
    # Paramètres de résumé
    MIN_SENTENCE_LENGTH = 10  # Caractères minimum
    SIMILARITY_THRESHOLD = 0.1  # Seuil pour les arêtes du graphe
    
    # MMR
    MMR_LAMBDA = 0.7  # Balance pertinence/diversité (0.5-0.8)
    
    # TextRank
    TEXTRANK_DAMPING = 0.85  # Facteur d'amortissement PageRank
    
    # ==========================================================================
    # INTELLIGENT NODE WEIGHTING (Domain-Aware Enhancements)
    # ==========================================================================
    
    # Arabic Discourse Markers - Pivot/result words that signal important content
    ARABIC_DISCOURSE_MARKERS = [
        # Contrast/Pivot markers
        "وبالتالي",      # consequently
        "نتيجة لذلك",    # as a result
        "لكن",           # but
        "ومع ذلك",       # however
        "لذلك",          # therefore
        "بسبب",          # because of
        "رغم",           # despite
        "إلا أن",        # except that
        "في المقابل",    # in contrast
        # Additive markers
        "علاوة على ذلك", # furthermore
        "من ناحية أخرى", # on the other hand
        "بالإضافة إلى",  # in addition to
        "وكذلك",         # and also
        "كما أن",        # also
        # Definitional markers
        "أي أن",         # that is to say
        "أي أنه",        # meaning that
        "تعني",          # means
        "ترمز",          # symbolizes
        "يُقصد",         # is meant
        "وهي",           # which is
        "وهو",           # which is (masc)
        # Announcement/citation markers
        "أكد",           # confirmed
        "أعلن",          # announced
        "كشف",           # revealed
        "أوضح",          # clarified
        "قال",           # said
        "صرح",           # stated
        # Exemplification markers
        "من أشهر",       # among the most famous
        "من أبرز",       # among the most prominent
        "مثل",           # such as
        "على سبيل المثال", # for example
    ]
    
    # Bonus Weights for Intelligent Scoring
    POSITION_BONUS_FIRST = 0.3   # Bonus for first 2 sentences
    POSITION_BONUS_LAST = 0.2    # Bonus for last sentence
    DISCOURSE_MARKER_BONUS = 0.5 # Bonus per discourse marker found
    ENTITY_BONUS = 0.2           # Bonus per named entity in sentence


# =============================================================================
# MODULE 2 : EXTRACTION DES EMBEDDINGS
# =============================================================================

class ArabicEmbeddingExtractor:
    """Extraction des embeddings de phrases avec AraBERT"""
    
    def __init__(self, model_name=ArabicExtractiveConfig.MODEL_NAME):
        """
        Initialise le tokenizer et le modèle AraBERT
        
        Args:
            model_name: Nom du modèle HuggingFace
        """
        print(f"📥 Chargement du modèle {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Mode évaluation
        print("✅ Modèle chargé avec succès!")
    
    def get_sentence_embeddings(self, sentences):
        """
        Calcule les embeddings pour une liste de phrases
        
        Args:
            sentences: Liste de phrases (strings)
            
        Returns:
            numpy.ndarray: Matrice d'embeddings [n_sentences, 768]
        """
        embeddings = []
        
        print(f"🔄 Extraction des embeddings pour {len(sentences)} phrases...")
        
        with torch.no_grad():  # Pas de gradients nécessaires
            for sentence in sentences:
                # Tokenization
                inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Mean pooling sur les tokens (ignore [CLS] et [SEP])
                token_embeddings = outputs.last_hidden_state[0]  # [seq_len, 768]
                sentence_embedding = token_embeddings.mean(dim=0)  # [768]
                
                embeddings.append(sentence_embedding.numpy())
        
        embeddings_matrix = np.array(embeddings)
        print(f"✅ Embeddings extraits : shape {embeddings_matrix.shape}")
        
        return embeddings_matrix


# =============================================================================
# MODULE 3 : SEGMENTATION DES PHRASES
# =============================================================================

# class ArabicSentenceSegmenter:
#     """Segmentation de textes arabes en phrases"""
    
#     def __init__(self):
#         """Initialise le pipeline spaCy pour l'arabe"""
#         print("📥 Chargement du pipeline spaCy pour l'arabe...")
#         self.nlp = Arabic()
#         self.nlp.add_pipe("sentencizer")
#         print("✅ Pipeline spaCy prêt!")
    
#     def split_into_sentences(self, text):
#         """
#         Découpe un texte en phrases
        
#         Args:
#             text: Texte arabe (string)
            
#         Returns:
#             list: Liste de phrases filtrées
#         """
#         # Créer un Doc spaCy
#         # --- Pre-processing for better segmentation ---

#         # 2. Replace multiple spaces/newlines with a single space
#         cleaned_text = re.sub(r'[\s\n]+', ' ', text)

#         # 3. Handle the common "Dr." abbreviation issue (د .)
#         cleaned_text = cleaned_text.replace("د .", "د.")
#         # Note: Keep the single space after the period to ensure the sentencizer works if "د." ends a sentence
#         cleaned_text = cleaned_text.replace("د.", "د. ") 

#         # 4. CRITICAL NEW STEP: Ensure a space follows every period (that isn't already followed by one)
#         # This pattern looks for a period (.), followed by a character (.), and inserts a space
#         # It aims to split run-on sentences like "sentence.New sentence"
#         # We exclude the cases where the character following the period is a digit (for years/numbers)
#         cleaned_text = re.sub(r'\.(?=[^\s\d])', '. ', cleaned_text)
#         doc = self.nlp(cleaned_text)
        
#         # Extraire les phrases
#         sentences = [sent.text.strip() for sent in doc.sents]
        
#         # Filtrer les phrases trop courtes
#         sentences = [
#             s for s in sentences 
#             if len(s) > ArabicExtractiveConfig.MIN_SENTENCE_LENGTH
#         ]
        
#         print(f"📄 Segmentation : {len(sentences)} phrases détectées")
        
#         return sentences


class ArabicSentenceSegmenter:
    """Segmentation de textes arabes en phrases"""
    
    def __init__(self):
        print("📥 Chargement du pipeline spaCy pour l'arabe...")
        # Use a blank model, then add the sentencizer
        self.nlp = Arabic()
        self.nlp.add_pipe("sentencizer")
        print("✅ Pipeline spaCy prêt!")
    
    def split_into_sentences(self, text: str) -> list:
        # 1. NOISE REDUCTION: Remove structural noise lines (like "اقرأ أيضاً:")
        # The 're.MULTILINE' flag (re.M) is crucial here.
        # This removes the entire line, ensuring it doesn't pollute the summary.
        text_without_noise = re.sub(r'^.*اقرأ أيضاً:.*$', '', text, flags=re.M)
        
        # 2. PARAGRAPH SEGMENTATION (Critical New Step)
        # Split the text into paragraphs based on one or more empty lines.
        # This preserves the structural breaks, treating each paragraph as a distinct block.
        paragraph_blocks = [
            block.strip() for block in re.split(r'\n{2,}', text_without_noise)
            if block.strip() # Filter out empty strings
        ]
        
        # Initialize final sentence list
        final_sentences = []
        
        for block in paragraph_blocks:
            # --- Pre-processing for better segmentation within the BLOCK ---

            # A. Normalize all remaining internal whitespace (spaces/single newlines) to a single space
            # We do this *inside* the block because we assume single newlines are soft wraps.
            cleaned_block = re.sub(r'[\s\n]+', ' ', block)

            # B. Handle the common "Dr." abbreviation issue (د .) and similar
            cleaned_block = cleaned_block.replace("د .", "د.")
            # Note: Keep the single space after the period to ensure the sentencizer works if "د." ends a sentence
            cleaned_block = cleaned_block.replace("د.", "د. ") 
            
            # C. Ensure a space follows every period (that isn't already followed by one)
            # This is your existing good logic, applied per block.
            cleaned_block = re.sub(r'\.(?=[^\s\d])', '. ', cleaned_block)

            # 3. SENTENCE SEGMENTATION (spaCy)
            doc = self.nlp(cleaned_block)
            
            # Extract sentences from the block
            sentences = [sent.text.strip() for sent in doc.sents]
            final_sentences.extend(sentences)
        
        # 4. Filter the entire list of sentences
        final_sentences = [
            s for s in final_sentences 
            if len(s) > ArabicExtractiveConfig.MIN_SENTENCE_LENGTH
        ]
        
        print(f"📄 Segmentation : {len(final_sentences)} phrases détectées")
        
        return final_sentences

# =============================================================================
# MODULE 4 : MATRICE DE SIMILARITÉ
# =============================================================================

def build_similarity_matrix(embeddings):
    """
    Calcule la matrice de similarité cosinus entre phrases
    
    Args:
        embeddings: Matrice d'embeddings [n_sentences, 768]
        
    Returns:
        numpy.ndarray: Matrice de similarité [n_sentences, n_sentences]
    """
    similarity_matrix = cosine_similarity(embeddings)
    
    print(f"🔢 Matrice de similarité calculée : shape {similarity_matrix.shape}")
    
    return similarity_matrix


# =============================================================================
# MODULE 5 : INTELLIGENT NODE WEIGHTING
# =============================================================================

def compute_sentence_bonuses(sentences, nlp=None):
    """
    Compute bonus weights for each sentence based on:
    - Position (first 2 + last sentence)
    - Arabic discourse markers
    - Named entity density (if nlp provided)
    
    Args:
        sentences: List of sentence strings
        nlp: Optional spaCy nlp object for NER
        
    Returns:
        dict: {sentence_index: bonus_score}
    """
    n_sentences = len(sentences)
    bonuses = {i: 0.0 for i in range(n_sentences)}
    
    config = ArabicExtractiveConfig
    
    for i, sentence in enumerate(sentences):
        # 1. Position Bias: Boost first 2 and last sentence
        if i < 2:
            bonuses[i] += config.POSITION_BONUS_FIRST
            print(f"📍 Position bonus (+{config.POSITION_BONUS_FIRST}) for sentence {i} (intro)")
        if i == n_sentences - 1:
            bonuses[i] += config.POSITION_BONUS_LAST
            print(f"📍 Position bonus (+{config.POSITION_BONUS_LAST}) for sentence {i} (conclusion)")
        
        # 2. Discourse Marker Bonus
        for marker in config.ARABIC_DISCOURSE_MARKERS:
            if marker in sentence:
                bonuses[i] += config.DISCOURSE_MARKER_BONUS
                print(f"💬 Discourse marker '{marker}' found in sentence {i} (+{config.DISCOURSE_MARKER_BONUS})")
                break  # Only count once per sentence
        
        # 3. Entity Density Bonus (if NLP available)
        if nlp is not None:
            try:
                doc = nlp(sentence)
                entity_count = len(doc.ents)
                if entity_count > 0:
                    entity_bonus = entity_count * config.ENTITY_BONUS
                    bonuses[i] += entity_bonus
                    print(f"🏷️ Entity bonus (+{entity_bonus:.2f}) for {entity_count} entities in sentence {i}")
            except Exception as e:
                print(f"⚠️ Entity extraction failed for sentence {i}: {e}")
    
    print(f"✅ Sentence bonuses computed: {bonuses}")
    return bonuses


# =============================================================================
# MODULE 6 : ALGORITHME TEXTRANK (Enhanced with Intelligent Weighting)
# =============================================================================

def textrank(similarity_matrix, sentences, top_n=3, sentence_bonuses=None):
    """
    Résumé extractif avec TextRank (basé sur PageRank)
    Enhanced with intelligent node weighting.
    
    Args:
        similarity_matrix: Matrice de similarité [n, n]
        sentences: Liste des phrases originales
        top_n: Nombre de phrases à extraire
        sentence_bonuses: Optional dict of {index: bonus_score} from compute_sentence_bonuses
        
    Returns:
        list: Phrases classées par importance
    """
    n_sentences = len(sentences)
    
    # Créer un graphe non-orienté
    graph = nx.Graph()
    graph.add_nodes_from(range(n_sentences))
    
    # Ajouter les arêtes (seulement si similarité > seuil)
    threshold = ArabicExtractiveConfig.SIMILARITY_THRESHOLD
    
    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            similarity = similarity_matrix[i][j]
            if similarity > threshold:
                graph.add_edge(i, j, weight=similarity)
    
    # Calculer les scores PageRank
    scores = nx.pagerank(
        graph, 
        alpha=ArabicExtractiveConfig.TEXTRANK_DAMPING,
        weight='weight'
    )
    
    # Apply intelligent bonuses to scores
    if sentence_bonuses:
        print("🧠 Applying intelligent node weighting...")
        for i in range(n_sentences):
            original_score = scores[i]
            bonus = sentence_bonuses.get(i, 0.0)
            scores[i] = original_score + bonus
            if bonus > 0:
                print(f"   Sentence {i}: {original_score:.4f} + {bonus:.2f} = {scores[i]:.4f}")
    
    # Trier les phrases par score décroissant
    ranked_sentences = sorted(
        [(scores[i], i, sentences[i]) for i in range(n_sentences)],
        reverse=True
    )
    
    # Sort selected sentences by their original index to preserve document flow
    top_indices = [idx for _, idx, _ in ranked_sentences[:top_n]]
    top_indices.sort()
    
    # Return sentences in original order
    top_sentences = [sentences[idx] for idx in top_indices]
    
    print(f"🏆 TextRank (Intelligent): {top_n} phrases sélectionnées (Ordre chronologique)")
    
    return top_sentences


# =============================================================================
# MODULE 7 : ALGORITHME MMR (Enhanced with Intelligent Weighting)
# =============================================================================

def mmr(embeddings, sentences, top_n=3, lambda_param=None, sentence_bonuses=None):
    """
    Résumé extractif avec MMR (Maximal Marginal Relevance)
    Maximise la pertinence ET la diversité
    Enhanced with intelligent node weighting.
    
    Args:
        embeddings: Matrice d'embeddings [n_sentences, 768]
        sentences: Liste des phrases originales
        top_n: Nombre de phrases à extraire
        lambda_param: Balance pertinence/diversité (défaut: config)
        sentence_bonuses: Optional dict of {index: bonus_score} from compute_sentence_bonuses
        
    Returns:
        list: Phrases diversifiées
    """
    if lambda_param is None:
        lambda_param = ArabicExtractiveConfig.MMR_LAMBDA
    
    n_sentences = len(sentences)
    
    # Embedding du document complet (moyenne des phrases)
    doc_embedding = embeddings.mean(axis=0, keepdims=True)
    
    # Calculer la pertinence de chaque phrase par rapport au document
    relevance_scores = cosine_similarity(embeddings, doc_embedding).flatten()
    
    # Apply intelligent bonuses to relevance scores
    if sentence_bonuses:
        print("🧠 MMR: Applying intelligent node weighting to relevance scores...")
        for i in range(n_sentences):
            bonus = sentence_bonuses.get(i, 0.0)
            if bonus > 0:
                original = relevance_scores[i]
                relevance_scores[i] = original + bonus
                print(f"   Sentence {i}: {original:.4f} + {bonus:.2f} = {relevance_scores[i]:.4f}")
    
    # Initialiser
    selected_indices = []
    remaining_indices = list(range(n_sentences))
    
    # Sélection itérative
    for _ in range(min(top_n, n_sentences)):
        mmr_scores = []
        
        for idx in remaining_indices:
            # Pertinence (now includes bonuses)
            relevance = relevance_scores[idx]
            
            # Redondance (similarité max avec les phrases déjà sélectionnées)
            if selected_indices:
                redundancy = max([
                    cosine_similarity(
                        embeddings[idx].reshape(1, -1),
                        embeddings[selected_idx].reshape(1, -1)
                    )[0][0]
                    for selected_idx in selected_indices
                ])
            else:
                redundancy = 0
            
            # Score MMR
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((mmr_score, idx))
        
        # Sélectionner la phrase avec le meilleur score MMR
        best_score, best_idx = max(mmr_scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    # Sort indices to preserve document flow
    selected_indices.sort()
    
    # Retourner les phrases sélectionnées dans l'ordre original
    selected_sentences = [sentences[idx] for idx in selected_indices]
    
    print(f"🎯 MMR (Intelligent): {len(selected_sentences)} phrases diversifiées sélectionnées (Ordre chronologique)")
    
    return selected_sentences


# =============================================================================
# MODULE 7 : PIPELINE COMPLET
# =============================================================================

class ArabicExtractiveSummarizer:
    """Pipeline complet de résumé extractif pour l'arabe"""
    
    def __init__(self):
        """Initialise tous les composants"""
        self.segmenter = ArabicSentenceSegmenter()
        self.embedding_extractor = ArabicEmbeddingExtractor()
    
    def summarize(self, text, method='textrank', top_n=3, lambda_param=None):
        """
        Génère un résumé extractif avec pondération intelligente
        
        Args:
            text: Texte arabe à résumer
            method: 'textrank' ou 'mmr'
            top_n: Nombre de phrases dans le résumé
            lambda_param: Pour MMR uniquement
            
        Returns:
            list: Phrases du résumé
        """
        print(f"\n{'='*60}")
        print(f"🚀 Résumé Extractif : Méthode {method.upper()} (Domain-Aware)")
        print(f"{'='*60}\n")
        
        # 1. Segmentation
        sentences = self.segmenter.split_into_sentences(text)
        
        if len(sentences) == 0:
            print("⚠️ Aucune phrase détectée!")
            return []
        
        if len(sentences) <= top_n:
            print(f"⚠️ Document trop court ({len(sentences)} phrases)")
            return sentences
        
        # 2. Extraction des embeddings
        embeddings = self.embedding_extractor.get_sentence_embeddings(sentences)
        
        # 3. Calcul de similarité
        similarity_matrix = build_similarity_matrix(embeddings)
        
        # 4. Compute intelligent sentence bonuses (Position, Discourse, Entities)
        # Note: NLP is set to None for now (entity bonus disabled for CPU perf)
        # Can be enabled by passing self.segmenter.nlp if an NER model is loaded
        sentence_bonuses = compute_sentence_bonuses(sentences, nlp=None)
        
        # 5. Sélection des phrases selon la méthode
        if method.lower() == 'textrank':
            summary = textrank(similarity_matrix, sentences, top_n, sentence_bonuses=sentence_bonuses)
        elif method.lower() == 'mmr':
            # MMR now uses intelligent bonuses too
            summary = mmr(embeddings, sentences, top_n, lambda_param, sentence_bonuses=sentence_bonuses)
        else:
            raise ValueError(f"Méthode inconnue: {method}. Utilisez 'textrank' ou 'mmr'")
        
        print(f"\n✅ Résumé généré avec succès!\n")
        
        return summary


# =============================================================================
# FONCTION UTILITAIRE
# =============================================================================

def print_summary(summary, title="RÉSUMÉ"):
    """Affiche joliment un résumé"""
    print(f"\n{'='*60}")
    print(f"📝 {title}")
    print(f"{'='*60}\n")
    
    for i, sentence in enumerate(summary, 1):
        print(f"{i}. {sentence}\n")
    
    print(f"{'='*60}\n")
