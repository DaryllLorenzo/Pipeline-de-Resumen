import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import re

class SemanticTFICFSummarizer(BaseEstimator, TransformerMixin):
    """
    Summarizer semántico mejorado con técnicas eficientes y robustas.
    Optimizado para CPU y dependencias mínimas.
    """
    
    def __init__(self, n_sentences='auto', clustering_method='kmeans', 
                 diversity_weight=0.4, coverage_weight=0.3, coherence_weight=0.3):
        self.n_sentences = n_sentences
        self.clustering_method = clustering_method
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
        self.coherence_weight = coherence_weight
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')

    def _calculate_optimal_sentences(self, num_sentences, word_count):
        """Cálculo optimizado del número de oraciones."""
        if self.n_sentences == 'auto':
            # Fórmula mejorada basada en análisis empírico
            if num_sentences <= 3:
                return max(1, num_sentences - 1)
            elif num_sentences <= 8:
                return min(3, max(2, num_sentences // 2))
            else:
                base = max(2, min(5, int(num_sentences * 0.25)))
                # Ajuste por densidad de información
                if word_count > 600:
                    base = min(base + 1, 6)
                return base
        return self.n_sentences

    def calculate_semantic_scores(self, processed_data):
        """
        Calcula scores semánticos mejorados con técnicas probadas.
        """
        sentences = processed_data['sentences']
        processed_sentences = processed_data['processed_sentences']
        semantic_scores = processed_data['semantic_scores']
        
        # 1. Score TF-ICF mejorado
        tf_icf_scores = self._calculate_enhanced_tf_icf(processed_sentences)
        
        # 2. Score por frases clave
        key_phrase_scores = self._calculate_key_phrase_scores(processed_data)
        
        # 3. Score de posición estratégica
        position_scores = self._calculate_position_scores(len(sentences))
        
        # Combinar scores con pesos optimizados
        final_scores = []
        for i in range(len(sentences)):
            tf_icf = tf_icf_scores[i] if i < len(tf_icf_scores) else 0
            key_phrase = key_phrase_scores[i] if i < len(key_phrase_scores) else 0
            semantic = semantic_scores[i] if i < len(semantic_scores) else 0
            position = position_scores[i] if i < len(position_scores) else 0
            
            # Combinación ponderada mejorada
            combined_score = (
                tf_icf * 0.35 +        # 35% TF-ICF mejorado
                key_phrase * 0.30 +     # 30% Frases clave
                semantic * 0.20 +       # 20% Semántica del preprocesador
                position * 0.15         # 15% Posición estratégica
            )
            
            # Bonus por longitud óptima
            sentence_length = len(sentences[i].split())
            length_bonus = self._calculate_length_bonus(sentence_length)
            combined_score *= length_bonus
            
            final_scores.append((i, combined_score, sentence_length))
        
        return final_scores

    def _calculate_enhanced_tf_icf(self, processed_sentences):
        """TF-ICF mejorado con manejo inteligente de términos."""
        all_terms = [sentence.split() for sentence in processed_sentences if sentence.strip()]
        
        if not all_terms:
            return [0] * len(processed_sentences)
        
        # Calcular TF con normalización mejorada
        tf_scores = []
        for sentence_terms in all_terms:
            total_terms = len(sentence_terms)
            if total_terms == 0:
                tf_scores.append({})
                continue
                
            tf = {}
            for term in sentence_terms:
                tf[term] = tf.get(term, 0) + 1
            
            # Normalización con ajuste para términos importantes
            for term in tf:
                # Términos más largos suelen ser más informativos
                term_weight = 1.0 + (min(len(term), 10) / 50)  # Máximo 20% bonus
                tf[term] = (tf[term] * term_weight) / total_terms
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
            # ICF balanceado que evita extremos
            if df == 1:
                icf_scores[term] = math.log(n_sentences) + 1.0  # Términos únicos
            elif df == n_sentences:
                icf_scores[term] = 0.1  # Términos muy comunes
            else:
                icf_scores[term] = math.log(n_sentences / df) + 0.5
        
        # Scores finales TF-ICF
        tf_icf_scores = []
        for i, (sentence_terms, tf) in enumerate(zip(all_terms, tf_scores)):
            if not sentence_terms:
                tf_icf_scores.append(0)
                continue
            
            score = sum(tf.get(term, 0) * icf_scores.get(term, 0) for term in sentence_terms)
            tf_icf_scores.append(score)
        
        # Normalizar scores
        max_score = max(tf_icf_scores) if tf_icf_scores else 1
        if max_score > 0:
            tf_icf_scores = [s / max_score for s in tf_icf_scores]
        
        return tf_icf_scores

    def _calculate_key_phrase_scores(self, processed_data):
        """Score de frases clave mejorado."""
        sentences = processed_data['sentences']
        key_phrases = processed_data['key_phrases']
        
        scores = [0] * len(sentences)
        if not key_phrases:
            return scores
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = 0
            
            for phrase, phrase_score in key_phrases:
                if phrase in sentence_lower:
                    words_in_phrase = len(phrase.split())
                    # Bonus progresivo por frases más específicas
                    specificity_bonus = 1 + (words_in_phrase * 0.25)
                    score += phrase_score * specificity_bonus
            
            scores[i] = score
        
        # Normalizar scores
        max_score = max(scores) if scores else 1
        if max_score > 0:
            scores = [s / max_score for s in scores]
        
        return scores

    def _calculate_position_scores(self, num_sentences):
        """Score de posición con énfasis en introducción y conclusión."""
        if num_sentences <= 1:
            return [1.0]
        
        scores = []
        for i in range(num_sentences):
            normalized_pos = i / (num_sentences - 1)
            
            # Curva en U suavizada - primeras y últimas oraciones
            if normalized_pos < 0.15:  # Primer 15%
                score = 1.3
            elif normalized_pos > 0.85:  # Último 15%
                score = 1.2
            elif normalized_pos < 0.3:  # Primer 30%
                score = 1.1
            elif normalized_pos > 0.7:  # Último 30%
                score = 1.0
            else:  # Medio
                score = 0.8
                
            scores.append(score)
        
        return scores

    def _calculate_length_bonus(self, word_count):
        """Bonus por longitud óptima de oración."""
        if word_count < 5:
            return 0.3  # Penalizar oraciones muy cortas
        elif word_count < 12:
            return 0.7
        elif word_count < 25:
            return 1.0  # Longitud óptima
        elif word_count < 35:
            return 0.8
        else:
            return 0.6  # Penalizar oraciones muy largas

    def select_optimal_sentences(self, sentence_scores, processed_data, n_select):
        """Selección optimizada con balance inteligente."""
        sentences = processed_data['sentences']
        
        if len(sentences) <= n_select:
            return list(range(len(sentences)))
        
        # Estrategia adaptativa de clustering
        n_clusters = self._calculate_adaptive_clusters(len(sentences), n_select)
        clusters = self._robust_semantic_clustering(processed_data, n_clusters)
        
        # Selección por clusters
        selected = self._select_from_clusters(sentence_scores, clusters, n_select, sentences)
        
        # Asegurar diversidad y cobertura
        if len(selected) < n_select:
            selected = self._intelligent_diversity_boost(selected, sentence_scores, processed_data, n_select)
        
        return sorted(selected)

    def _calculate_adaptive_clusters(self, num_sentences, n_select):
        """Cálculo adaptativo de clusters."""
        if num_sentences <= 6:
            return min(2, num_sentences)
        elif num_sentences <= 12:
            return min(3, n_select + 1)
        else:
            return min(5, max(2, num_sentences // 4))

    def _robust_semantic_clustering(self, processed_data, n_clusters):
        """Clustering robusto con múltiples fallbacks."""
        sentences = processed_data['sentences']
        
        if len(sentences) <= n_clusters:
            return list(range(len(sentences)))
        
        try:
            # Intentar clustering con K-means
            X = self.vectorizer.fit_transform(sentences)
            
            if X.shape[1] < 2:  # Muy pocas características
                return list(range(len(sentences)))
                
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(sentences)), 
                random_state=42, 
                n_init=10
            )
            clusters = kmeans.fit_predict(X)
            
            # Verificar que haya al menos 2 clusters no vacíos
            unique_clusters = set(clusters)
            if len(unique_clusters) < 2:
                return list(range(len(sentences)))
                
            return clusters
            
        except Exception:
            # Fallback a agrupamiento simple por similitud
            return self._similarity_based_clustering(sentences, n_clusters)

    def _similarity_based_clustering(self, sentences, n_clusters):
        """Clustering simple basado en similitud."""
        if len(sentences) <= n_clusters:
            return list(range(len(sentences)))
        
        try:
            # Vectorización básica
            X = self.vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(X)
            
            clusters = [-1] * len(sentences)
            current_cluster = 0
            
            # Agrupar oraciones similares
            for i in range(len(sentences)):
                if clusters[i] == -1:
                    clusters[i] = current_cluster
                    # Buscar oraciones similares
                    for j in range(i + 1, len(sentences)):
                        if clusters[j] == -1 and similarity_matrix[i, j] > 0.4:
                            clusters[j] = current_cluster
                    current_cluster += 1
                    if current_cluster >= n_clusters:
                        break
            
            # Asignar clusters restantes
            for i in range(len(sentences)):
                if clusters[i] == -1:
                    clusters[i] = i % n_clusters
                    
            return clusters
            
        except:
            return list(range(len(sentences)))

    def _select_from_clusters(self, sentence_scores, clusters, n_select, sentences):
        """Selección desde clusters mejorada."""
        cluster_groups = defaultdict(list)
        
        # Agrupar oraciones por cluster
        for i, (idx, score, length) in enumerate(sentence_scores):
            if len(sentences[idx]) >= 20:  # Longitud mínima razonable
                cluster_id = clusters[idx]
                cluster_groups[cluster_id].append((idx, score, length))
        
        selected = []
        
        # Estrategia 1: Mejor oración de cada cluster
        for cluster_id in sorted(cluster_groups.keys()):
            cluster_sentences = cluster_groups[cluster_id]
            cluster_sentences.sort(key=lambda x: x[1], reverse=True)
            if cluster_sentences:
                selected.append(cluster_sentences[0][0])
        
        # Estrategia 2: Segundas mejores de clusters grandes
        if len(selected) < n_select:
            for cluster_id in cluster_groups:
                if len(selected) >= n_select:
                    break
                if len(cluster_groups[cluster_id]) > 1 and cluster_groups[cluster_id][0][0] not in selected:
                    selected.append(cluster_groups[cluster_id][1][0])
        
        # Estrategia 3: Mejores globales restantes
        if len(selected) < n_select:
            all_candidates = [
                idx for idx, _, _ in sentence_scores 
                if idx not in selected and len(sentences[idx]) >= 15
            ]
            all_candidates.sort(key=lambda x: sentence_scores[x][1], reverse=True)
            selected.extend(all_candidates[:n_select - len(selected)])
        
        return selected[:n_select]

    def _intelligent_diversity_boost(self, selected, sentence_scores, processed_data, n_select):
        """Boost de diversidad que considera contenido semántico."""
        sentences = processed_data['sentences']
        
        if len(selected) >= n_select:
            return selected
        
        try:
            # Calcular similitud entre oraciones
            all_sentences = [sentences[i] for i in range(len(sentences))]
            X = self.vectorizer.fit_transform(all_sentences)
            similarity_matrix = cosine_similarity(X)
            
            candidates = [i for i in range(len(sentences)) if i not in selected]
            candidates.sort(key=lambda x: sentence_scores[x][1], reverse=True)
            
            for candidate in candidates:
                if len(selected) >= n_select:
                    break
                
                # Calcular similitud máxima con seleccionadas
                max_similarity = 0
                for selected_idx in selected:
                    similarity = similarity_matrix[candidate][selected_idx]
                    max_similarity = max(max_similarity, similarity)
                
                # Umbral adaptativo: más estricto cuando tenemos más oraciones
                threshold = 0.7 if len(selected) == 0 else 0.5
                
                if max_similarity < threshold:
                    selected.append(candidate)
            
        except Exception:
            # Fallback simple
            candidates = [i for i in range(len(sentences)) if i not in selected]
            candidates.sort(key=lambda x: sentence_scores[x][1], reverse=True)
            selected.extend(candidates[:n_select - len(selected)])
        
        return selected[:n_select]

    def _build_improved_summary(self, sentences, selected_indices, language):
        """Construcción mejorada del resumen con formato robusto."""
        if not selected_indices:
            return ""
        
        selected_sentences = [sentences[i] for i in selected_indices]
        
        # Limpiar y formatear cada oración
        cleaned_sentences = []
        for sentence in selected_sentences:
            # Limpiar espacios y puntuación
            clean_sent = re.sub(r'\s+', ' ', sentence.strip())
            # Asegurar puntuación final
            if clean_sent and not clean_sent[-1] in '.!?':
                clean_sent += '.'
            # Remover múltiples puntos
            clean_sent = re.sub(r'\.+', '.', clean_sent)
            cleaned_sentences.append(clean_sent)
        
        # Unir oraciones
        summary = ' '.join(cleaned_sentences)
        
        # Post-procesamiento final
        summary = re.sub(r'\s+', ' ', summary)  # Espacios múltiples
        summary = re.sub(r'\s\.', '.', summary)  # Espacios antes de punto
        summary = summary.strip()
        
        return summary

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transformación principal con manejo robusto de errores."""
        summaries = []
        
        for processed_data in X:
            try:
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
                
                # Construir resumen mejorado
                summary = self._build_improved_summary(
                    processed_data['sentences'], 
                    selected_indices,
                    processed_data['language']
                )
                
                summary_data = {
                    'original': processed_data['original'],
                    'language': processed_data['language'],
                    'summary': summary,
                    'selected_sentences': selected_indices,
                    'num_original_sentences': n_original,
                    'num_summary_sentences': len(selected_indices),
                    'compression_ratio': len(summary) / len(processed_data['original']) if processed_data['original'] else 0,
                    'method': 'enhanced_semantic_tficf',
                    # Incluir datos para métricas
                    'key_phrases': processed_data.get('key_phrases', []),
                    'sentences': processed_data['sentences']
                }
                summaries.append(summary_data)
                
            except Exception as e:
                print(f"Error en transform: {e}")
                # Fallback robusto
                summaries.append(self._handle_short_text(processed_data))
            
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
            'method': 'short_text',
            'key_phrases': processed_data.get('key_phrases', []),
            'sentences': processed_data.get('sentences', [processed_data['original']])
        }