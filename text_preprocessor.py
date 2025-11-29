import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Configuración de NLTK
try:
    nltk.data.find('tokenizers/punkt') #nltk.data.find('tokenizers/punkt_tab') # 
except LookupError:
    nltk.download('punkt_tab')

class EnhancedTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesador de texto avanzado para español e inglés.
    """
    
    def __init__(self, min_word_length=3, use_semantic_analysis=True):
        self.min_word_length = min_word_length
        self.use_semantic_analysis = use_semantic_analysis
        
        # Stopwords expandidas
        self.stopwords_es = {
            'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 
            'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 
            'es', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 
            'este', 'esta', 'está', 'son', 'ser', 'fue', 'ha', 'han', 
            'si', 'sí', 'sino', 'algo', 'todo', 'tan', 'así', 'mi', 
            'porque', 'muy', 'sin', 'sobre', 'me', 'ya', 'cuando', 'donde',
            'cómo', 'qué', 'quién', 'cual', 'cuyo', 'cuyas', 'cuyos',
            'también', 'etc', 'entre', 'hasta', 'desde', 'durante', 'mediante'
        }
        
        self.stopwords_en = {
            'the', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that',
            'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
            'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 
            'by', 'but', 'not', 'what', 'all', 'were', 'when', 'which',
            'there', 'their', 'been', 'has', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'we', 'our', 'she',
            'also', 'etc', 'between', 'through', 'during', 'using'
        }

    def detect_language(self, text):
        """Detección mejorada de idioma."""
        spanish_chars = re.findall(r'[áéíóúñÁÉÍÓÚÑ]', text)
        if spanish_chars:
            return 'es'
        
        spanish_common = {'el', 'la', 'de', 'que', 'y', 'en', 'los', 'las', 'del', 'se'}
        english_common = {'the', 'and', 'of', 'to', 'in', 'is', 'for', 'on', 'are', 'with'}
        
        words = text.lower().split()
        spanish_score = sum(1 for w in words if w in spanish_common)
        english_score = sum(1 for w in words if w in english_common)
        
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        length_bonus = 1 if avg_word_length > 5 else -1
        
        final_spanish_score = spanish_score + length_bonus
        final_english_score = english_score - length_bonus
        
        return 'es' if final_spanish_score > final_english_score else 'en'

    def split_sentences(self, text):
        """División de oraciones con NLTK."""
        try:
            sentences = nltk.tokenize.sent_tokenize(text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]

    def extract_key_phrases(self, text, language, top_n=10):
        """Extrae frases clave usando TF-IDF."""
        try:
            stop_words = list(self.stopwords_en if language == 'en' else self.stopwords_es)
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=top_n, stop_words=stop_words)
            X = vectorizer.fit_transform([text])
            features = vectorizer.get_feature_names_out()
            scores = X.toarray()[0]
            
            key_phrases = [(features[i], scores[i]) for i in range(len(features))]
            return sorted(key_phrases, key=lambda x: x[1], reverse=True)[:top_n]
        except:
            return []

    def preprocess_text(self, text, language):
        """Preprocesamiento con preservación de contexto."""
        text = re.sub(r'[^\w\sáéíóúñÁÉÍÓÚÑ]', ' ', text.lower())
        words = text.split()
        
        stopwords = self.stopwords_es if language == 'es' else self.stopwords_en
        words = [w for w in words if w not in stopwords and len(w) >= self.min_word_length]
        
        return ' '.join(words)

    def _calculate_semantic_importance(self, text, sentences, key_phrases):
        """Calcula importancia semántica basada en frases clave."""
        scores = []
        key_phrases_dict = dict(key_phrases)
        
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            
            for phrase, phrase_score in key_phrases:
                if phrase in sentence_lower:
                    score += phrase_score * len(phrase.split())
            
            # Bonus por posición
            if i == 0 or i == len(sentences) - 1:
                score *= 1.3
                
            # Bonus por longitud
            length_bonus = min(len(sentence.split()) / 25, 0.5)
            score *= (1 + length_bonus)
            
            scores.append(score)
        
        return scores

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transformación con análisis semántico."""
        results = []
        for text in X:
            if not text or len(text.strip()) < 100:
                continue
                
            language = self.detect_language(text)
            sentences = self.split_sentences(text)
            
            if len(sentences) < 2:
                sentences = [text]
                
            processed_sentences = [self.preprocess_text(s, language) for s in sentences]
            key_phrases = self.extract_key_phrases(text, language)
            semantic_scores = self._calculate_semantic_importance(text, sentences, key_phrases)
            
            results.append({
                'original': text,
                'language': language,
                'sentences': sentences,
                'processed_sentences': processed_sentences,
                'num_sentences': len(sentences),
                'word_count': len(text.split()),
                'key_phrases': key_phrases,
                'semantic_scores': semantic_scores,
                'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            })
        return results