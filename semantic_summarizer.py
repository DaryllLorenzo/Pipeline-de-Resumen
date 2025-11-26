import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict

class SemanticTFICFSummarizer(BaseEstimator, TransformerMixin):
    """
    Summarizer semántico mejorado con métricas avanzadas.
    """
    
    def __init__(self, n_sentences='auto', clustering_method='kmeans', 
                 diversity_weight=0.4, coverage_weight=0.3, coherence_weight=0.3):
        self.n_sentences = n_sentences
        self.clustering_method = clustering_method
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
        self.coherence_weight = coherence_weight
        self.vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')

    def _calculate_optimal_sentences(self, num_sentences, word_count):
        """Cálculo optimizado del número de oraciones."""
        if self.n_sentences == 'auto':
            base = max(2, min(6, int(num_sentences * 0.2)))
            if word_count > 800:
                base += 1
            if word_count > 1200:
                base += 1
            return base
        return self.n_sentences

    def calculate_semantic_scores(self, processed_data):
        """
        Calcula scores semánticos combinando TF-ICF, frases clave y posición.
        """
        sentences = processed_data['sentences']
        processed_sentences = processed_data['processed_sentences']
        semantic_scores = processed_data['semantic_scores']
        
        # 1. Score TF-ICF tradicional
        tf_icf_scores = self._calculate_tf_icf_scores(processed_sentences)
        
        # 2. Score por frases clave
        key_phrase_scores = self._calculate_key_phrase_scores(processed_data)
        
        # 3. Combinar scores con pesos optimizados
        final_scores = []
        for i in range(len(sentences)):
            tf_icf = tf_icf_scores[i] if i < len(tf_icf_scores) else 0
            key_phrase = key_phrase_scores[i] if i < len(key_phrase_scores) else 0
            semantic = semantic_scores[i] if i < len(semantic_scores) else 0
            
            # Combinación ponderada
            combined_score = (
                tf_icf * 0.4 +        # 40% TF-ICF tradicional
                key_phrase * 0.3 +     # 30% Frases clave
                semantic * 0.3         # 30% Semántica y posición
            )
            
            final_scores.append((i, combined_score, len(sentences[i].split())))
        
        return final_scores

    def _calculate_tf_icf_scores(self, processed_sentences):
        """Implementación mejorada de TF-ICF."""
        all_terms = [sentence.split() for sentence in processed_sentences if sentence.strip()]
        
        if not all_terms:
            return [0] * len(processed_sentences)
        
        # Calcular TF
        tf_scores = []
        for sentence_terms in all_terms:
            total_terms = len(sentence_terms)
            if total_terms == 0:
                tf_scores.append({})
                continue
                
            tf = {}
            for term in sentence_terms:
                tf[term] = tf.get(term, 0) + 1
            
            for term in tf:
                tf[term] = tf[term] / total_terms
            tf_scores.append(tf)
        
        # Calcular ICF mejorado
        n_sentences = len(all_terms)
        doc_freq = {}
        
        for sentence_terms in all_terms:
            unique_terms = set(sentence_terms)
            for term in unique_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        icf_scores = {}
        for term, df in doc_freq.items():
            icf_scores[term] = math.log((n_sentences + 1) / (df + 0.5)) + 1
        
        # Scores finales TF-ICF
        tf_icf_scores = []
        for i, (sentence_terms, tf) in enumerate(zip(all_terms, tf_scores)):
            if not sentence_terms:
                tf_icf_scores.append(0)
                continue
            
            score = sum(tf.get(term, 0) * icf_scores.get(term, 0) for term in sentence_terms)
            tf_icf_scores.append(score)
        
        return tf_icf_scores

    def _calculate_key_phrase_scores(self, processed_data):
        """Calcula scores basados en frases clave."""
        sentences = processed_data['sentences']
        key_phrases = processed_data['key_phrases']
        
        scores = [0] * len(sentences)
        key_phrases_dict = dict(key_phrases)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = 0
            
            for phrase, phrase_score in key_phrases:
                if phrase in sentence_lower:
                    words_in_phrase = len(phrase.split())
                    score += phrase_score * (1 + words_in_phrase * 0.2)
            
            scores[i] = score
        
        # Normalizar scores
        max_score = max(scores) if scores else 1
        if max_score > 0:
            scores = [s / max_score for s in scores]
        
        return scores

    def select_optimal_sentences(self, sentence_scores, processed_data, n_select):
        """Selección optimizada balanceando relevancia y diversidad."""
        sentences = processed_data['sentences']
        
        if len(sentences) <= n_select:
            return list(range(len(sentences)))
        
        # Aplicar clustering para diversidad
        n_clusters = min(n_select + 1, len(sentences) // 2)
        clusters = self._semantic_clustering(processed_data, n_clusters)
        
        # Selección por clusters
        selected = self._select_from_clusters(sentence_scores, clusters, n_select, sentences)
        
        # Asegurar diversidad temática
        if len(selected) < n_select:
            selected = self._diversity_boost(selected, sentence_scores, processed_data, n_select)
        
        return sorted(selected)

    def _semantic_clustering(self, processed_data, n_clusters):
        """Clustering semántico mejorado."""
        sentences = processed_data['sentences']
        
        if len(sentences) <= n_clusters:
            return list(range(len(sentences)))
        
        try:
            X = self.vectorizer.fit_transform(sentences)
            
            if X.shape[1] == 0:
                return list(range(len(sentences)))
                
            kmeans = KMeans(n_clusters=min(n_clusters, X.shape[0]), random_state=42, n_init=10)
            return kmeans.fit_predict(X)
        except:
            return list(range(len(sentences)))

    def _select_from_clusters(self, sentence_scores, clusters, n_select, sentences):
        """Selección desde clusters con criterios múltiples."""
        cluster_groups = defaultdict(list)
        
        for i, (idx, score, length) in enumerate(sentence_scores):
            if len(sentences[idx]) >= 25:  # Longitud mínima
                cluster_id = clusters[idx]
                cluster_groups[cluster_id].append((idx, score, length))
        
        # Seleccionar mejor oración de cada cluster
        selected = []
        for cluster_id in cluster_groups:
            cluster_groups[cluster_id].sort(key=lambda x: x[1], reverse=True)
            if cluster_groups[cluster_id]:
                selected.append(cluster_groups[cluster_id][0][0])
        
        # Completar con las mejores globales si es necesario
        if len(selected) < n_select:
            all_candidates = [idx for idx, _, _ in sentence_scores 
                            if idx not in selected and len(sentences[idx]) >= 25]
            all_candidates.sort(key=lambda x: sentence_scores[x][1], reverse=True)
            selected.extend(all_candidates[:n_select - len(selected)])
        
        return selected[:n_select]

    def _diversity_boost(self, selected, sentence_scores, processed_data, n_select):
        """Boost de diversidad usando similitud semántica."""
        sentences = processed_data['sentences']
        
        if len(selected) >= n_select:
            return selected
        
        # Calcular similitud entre oraciones seleccionadas y candidatas
        try:
            all_sentences = [sentences[i] for i in range(len(sentences))]
            similarity_matrix = cosine_similarity(
                self.vectorizer.fit_transform(all_sentences)
            )
            
            candidates = [i for i in range(len(sentences)) if i not in selected]
            candidates.sort(key=lambda x: sentence_scores[x][1], reverse=True)
            
            for candidate in candidates:
                if len(selected) >= n_select:
                    break
                
                # Calcular similitud máxima con oraciones ya seleccionadas
                max_similarity = 0
                for selected_idx in selected:
                    similarity = similarity_matrix[candidate][selected_idx]
                    max_similarity = max(max_similarity, similarity)
                
                # Si es suficientemente diferente, agregar
                if max_similarity < 0.6:  # Umbral de similitud
                    selected.append(candidate)
            
        except:
            # Fallback: agregar las mejores restantes
            candidates = [i for i in range(len(sentences)) if i not in selected]
            candidates.sort(key=lambda x: sentence_scores[x][1], reverse=True)
            selected.extend(candidates[:n_select - len(selected)])
        
        return selected[:n_select]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transformación principal del summarizer."""
        summaries = []
        
        for processed_data in X:
            n_original = processed_data['num_sentences']
            if n_original <= 1:
                summary_data = self._handle_short_text(processed_data)
                summaries.append(summary_data)
                continue
            
            n_summary = self._calculate_optimal_sentences(
                n_original, 
                processed_data['word_count']
            )
            
            # Calcular scores semánticos
            sentence_scores = self.calculate_semantic_scores(processed_data)
            
            # Selección optimizada
            selected_indices = self.select_optimal_sentences(
                sentence_scores, processed_data, n_summary
            )
            
            # Construir resumen
            summary_sentences = [processed_data['sentences'][i] for i in selected_indices]
            summary = '. '.join(summary_sentences) + '.'
            
            summary_data = {
                'original': processed_data['original'],
                'language': processed_data['language'],
                'summary': summary,
                'selected_sentences': selected_indices,
                'num_original_sentences': n_original,
                'num_summary_sentences': len(selected_indices),
                'compression_ratio': len(summary) / len(processed_data['original']),
                'method': 'semantic_tficf_advanced',
                # Incluir datos necesarios para métricas
                'key_phrases': processed_data.get('key_phrases', []),
                'sentences': processed_data['sentences']  # Asegurar que esté disponible
            }
            summaries.append(summary_data)
            
        return summaries

    def _handle_short_text(self, processed_data):
        """Manejo de textos cortos."""
        return {
            'original': processed_data['original'],
            'language': processed_data['language'],
            'summary': processed_data['original'],
            'selected_sentences': [0],
            'num_original_sentences': 1,
            'num_summary_sentences': 1,
            'compression_ratio': 1.0,
            'method': 'short_text'
        }