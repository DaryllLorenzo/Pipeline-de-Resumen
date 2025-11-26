import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesador de texto para español e inglés.
    Divide el texto en oraciones y detecta el idioma.
    """
    
    def __init__(self):
        self.stopwords_es = {'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 
                           'con', 'no', 'una', 'su', 'al', 'es', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o'}
        self.stopwords_en = {'the', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on',
                           'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or'}

    def detect_language(self, text):
        """Detecta si el texto es español o inglés basado en caracteres especiales."""
        spanish_chars = re.findall(r'[áéíóúñÁÉÍÓÚÑ]', text)
        return 'es' if len(spanish_chars) > 0 else 'en'

    def split_sentences(self, text):
        """Divide el texto en oraciones usando expresiones regulares."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def preprocess_text(self, text, language):
        """Limpia y tokeniza el texto según el idioma."""
        # Convertir a minúsculas y eliminar caracteres no alfabéticos
        text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s]', '', text.lower())
        words = text.split()
        
        # Filtrar stopwords según idioma
        stopwords = self.stopwords_es if language == 'es' else self.stopwords_en
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        return ' '.join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Procesa el texto y devuelve oraciones preprocesadas."""
        results = []
        for text in X:
            language = self.detect_language(text)
            sentences = self.split_sentences(text)
            processed_sentences = [self.preprocess_text(s, language) for s in sentences]
            results.append({
                'original': text,
                'language': language,
                'sentences': sentences,
                'processed_sentences': processed_sentences
            })
        return results

class TFICFSummarizer(BaseEstimator, TransformerMixin):
    """
    Implementación del algoritmo TF-ICF para resumen extractivo.
    Calcula la importancia de las oraciones usando Term Frequency - Inverse Class Frequency.
    """
    
    def __init__(self, n_sentences=2):
        self.n_sentences = n_sentences
        self.vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b\w+\b')

    def calculate_tf(self, sentence_terms):
        """Calcula la frecuencia de términos (TF) para una oración."""
        total_terms = len(sentence_terms)
        if total_terms == 0:
            return {}
        
        tf = {}
        for term in sentence_terms:
            tf[term] = tf.get(term, 0) + 1
        
        # Normalizar
        for term in tf:
            tf[term] = tf[term] / total_terms
            
        return tf

    def calculate_icf(self, all_sentences_terms):
        """Calcula la frecuencia inversa de clase (ICF) para los términos."""
        n_sentences = len(all_sentences_terms)
        icf = {}
        
        for sentence_terms in all_sentences_terms:
            unique_terms = set(sentence_terms)
            for term in unique_terms:
                icf[term] = icf.get(term, 0) + 1
        
        # Calcular ICF (log(N/n))
        for term in icf:
            icf[term] = np.log(n_sentences / icf[term])
            
        return icf

    def calculate_sentence_scores(self, processed_data):
        """Calcula los puntajes TF-ICF para cada oración."""
        all_sentences = processed_data['processed_sentences']
        all_terms = [sentence.split() for sentence in all_sentences]
        
        # Calcular ICF para todos los términos
        icf = self.calculate_icf(all_terms)
        
        sentence_scores = []
        for i, sentence_terms in enumerate(all_terms):
            tf = self.calculate_tf(sentence_terms)
            
            # Calcular puntaje TF-ICF para la oración
            score = 0
            for term in sentence_terms:
                if term in tf and term in icf:
                    score += tf[term] * icf[term]
            
            sentence_scores.append((i, score, len(sentence_terms)))
        
        return sentence_scores

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Genera resúmenes para cada texto en X."""
        summaries = []
        for processed_data in X:
            sentence_scores = self.calculate_sentence_scores(processed_data)
            
            # Filtrar oraciones vacías y ordenar por puntaje
            valid_scores = [(i, score) for i, score, length in sentence_scores if length > 0]
            valid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Seleccionar las mejores oraciones
            top_indices = [i for i, _ in valid_scores[:self.n_sentences]]
            top_indices.sort()  # Mantener orden original
            
            # Construir resumen
            summary_sentences = [processed_data['sentences'][i] for i in top_indices]
            summary = '. '.join(summary_sentences) + '.'
            
            summaries.append({
                'original': processed_data['original'],
                'language': processed_data['language'],
                'summary': summary,
                'selected_sentences': top_indices
            })
            
        return summaries

# Pipeline completo
summarization_pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('summarizer', TFICFSummarizer(n_sentences=2))
])

# Ejemplos de prueba
if __name__ == "__main__":
    # Textos de ejemplo
    spanish_text = """
    El aprendizaje automático es una rama de la inteligencia artificial. 
    Los algoritmos de machine learning permiten a las computadoras aprender patrones en los datos. 
    En la actualidad, el deep learning ha revolucionado muchas áreas. 
    España es un país con gran desarrollo en tecnología. 
    Los investigadores españoles contribuyen significativamente al campo.
    """
    
    english_text = """
    Machine learning is a branch of artificial intelligence. 
    Machine learning algorithms allow computers to learn patterns from data. 
    Currently, deep learning has revolutionized many areas. 
    The United States is a leader in technology development. 
    American researchers contribute significantly to the field.
    """

    # Procesar textos
    results = summarization_pipeline.fit_transform([spanish_text, english_text])
    
    # Mostrar resultados
    for i, result in enumerate(results):
        print(f"\n--- Texto {i+1} ({result['language'].upper()}) ---")
        print("Original:")
        print(result['original'][:100] + "...")
        print("\nResumen:")
        print(result['summary'])
        print(f"Oraciones seleccionadas: {result['selected_sentences']}")
        print("-" * 50)
