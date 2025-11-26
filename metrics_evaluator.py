import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

class MetricsEvaluator:
    """
    Evaluador de métricas para resúmenes, separado del summarizer para evitar recursión.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')

    def calculate_improved_bleu(self, original, summary):
        """BLEU score mejorado con múltiples referencias."""
        try:
            original_sentences = nltk.tokenize.sent_tokenize(original)
            
            if not original_sentences or not summary:
                return 0.0
                
            summary_tokens = summary.lower().split()
            references = [sent.lower().split() for sent in original_sentences[:3]]
            
            smoothie = SmoothingFunction().method1
            score = sentence_bleu(
                references, 
                summary_tokens,
                smoothing_function=smoothie,
                weights=(0.5, 0.3, 0.2, 0)
            )
            return min(score, 1.0)
        except:
            return 0.0

    def calculate_rouge_like(self, original, summary):
        """Métrica tipo ROUGE."""
        try:
            original_words = set(original.lower().split())
            summary_words = set(summary.lower().split())
            
            if not original_words:
                return 0.0
                
            overlapping = len(original_words.intersection(summary_words))
            recall = overlapping / len(original_words)
            precision = overlapping / len(summary_words) if summary_words else 0
            
            if recall + precision == 0:
                return 0.0
            return 2 * (recall * precision) / (recall + precision)
        except:
            return 0.0

    def calculate_semantic_coverage(self, processed_data, selected_indices):
        """Cobertura semántica basada en frases clave."""
        try:
            # Verificar que tenemos los datos necesarios
            if not processed_data or not isinstance(processed_data, dict):
                return 0.3
                
            key_phrases = processed_data.get('key_phrases', [])
            sentences = processed_data.get('sentences', [])
            
            if not key_phrases or not sentences or not selected_indices:
                return 0.3
                
            covered_phrases = 0
            selected_sentences = []
            
            # Obtener oraciones seleccionadas de forma segura
            for idx in selected_indices:
                if idx < len(sentences):
                    selected_sentences.append(sentences[idx].lower())
            
            if not selected_sentences:
                return 0.3
                
            # Contar frases clave cubiertas
            for phrase, _ in key_phrases[:8]:  # Usar top 8 frases
                phrase_lower = phrase.lower()
                for sentence in selected_sentences:
                    if phrase_lower in sentence:
                        covered_phrases += 1
                        break
            
            return covered_phrases / min(8, len(key_phrases))
            
        except Exception as e:
            print(f"⚠️  Error en cobertura semántica: {e}")
            return 0.3

    def calculate_coherence(self, summary):
        """Coherencia del resumen."""
        try:
            # Primero verificar que el summary tenga múltiples oraciones
            if not summary or len(summary.strip()) < 10:
                return 0.0
                
            sentences = nltk.tokenize.sent_tokenize(summary)
            if len(sentences) <= 1:
                return 1.0  # Un solo enunciado es coherente por defecto
                
            # Limpiar oraciones vacías
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            if len(sentences) <= 1:
                return 1.0
                
            # Vectorizar y calcular similitud
            vectors = self.vectorizer.fit_transform(sentences)
            if vectors.shape[0] <= 1:
                return 1.0
                
            similarities = []
            
            for i in range(len(sentences)-1):
                sim = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])
                if sim.size > 0:
                    similarities.append(sim[0][0])
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            print(f"⚠️  Error en cálculo de coherencia: {e}")
            return 0.5  # Valor por defecto en caso de error

    def calculate_diversity(self, text):
        """Diversidad léxica."""
        try:
            if not text:
                return 0.0
                
            words = text.lower().split()
            if len(words) <= 1:
                return 0.0
                
            unique_words = set(words)
            return len(unique_words) / len(words)
        except:
            return 0.0

    def calculate_redundancy(self, text):
        """Redundancia en el resumen."""
        try:
            if not text:
                return 0.0
                
            words = text.lower().split()
            if len(words) <= 1:
                return 0.0
                
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            repeated = sum(1 for count in word_freq.values() if count > 1)
            return repeated / len(words)
        except:
            return 0.0

    def calculate_all_metrics(self, original, summary, processed_data=None, selected_indices=None):
        """
        Calcula todas las métricas sin recursión.
        
        Args:
            original (str): Texto original
            summary (str): Resumen generado
            processed_data (dict): Datos procesados (opcional)
            selected_indices (list): Índices de oraciones seleccionadas (opcional)
            
        Returns:
            dict: Diccionario con todas las métricas
        """
        metrics = {
            'bleu_score': self.calculate_improved_bleu(original, summary),
            'rouge_like_score': self.calculate_rouge_like(original, summary),
            'diversity_score': self.calculate_diversity(summary),
            'coherence_score': self.calculate_coherence(summary),
            'redundancy_score': self.calculate_redundancy(summary),
        }
        
        # Métricas que requieren processed_data - con manejo robusto de errores
        if (processed_data is not None and selected_indices is not None and 
            isinstance(processed_data, dict) and isinstance(selected_indices, list)):
            metrics['coverage_score'] = self.calculate_semantic_coverage(processed_data, selected_indices)
        else:
            metrics['coverage_score'] = 0.3  # Valor por defecto
        
        # Calcular score general
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        return metrics

    def _calculate_overall_score(self, metrics):
        """Calcula el puntaje general basado en métricas individuales."""
        try:
            weights = {
                'rouge_like_score': 0.3,
                'bleu_score': 0.2,
                'coverage_score': 0.2,
                'coherence_score': 0.15,
                'diversity_score': 0.1,
                'redundancy_score': 0.05
            }
            
            overall = 0
            for metric, weight in weights.items():
                score = metrics.get(metric, 0)
                if metric == 'redundancy_score':
                    score = 1 - score  # Convertir redundancia en puntaje positivo
                overall += score * weight
            
            return min(max(overall, 0), 1)  # Asegurar entre 0 y 1
            
        except Exception as e:
            print(f"⚠️  Error calculando overall score: {e}")
            return 0.5


class AdvancedSummaryEvaluator:
    """
    Evaluador avanzado para comparar múltiples métodos.
    """
    
    def __init__(self):
        self.metrics_evaluator = MetricsEvaluator()
        self.metrics_history = []
    
    def comprehensive_evaluation(self, original, summary, method_name, processed_data=None, selected_indices=None):
        """Evaluación comprehensiva con múltiples dimensiones."""
        
        metrics = self.metrics_evaluator.calculate_all_metrics(
            original, summary, processed_data, selected_indices
        )
        
        evaluation = {
            'method': method_name,
            'compression_ratio': len(summary) / len(original) if original else 0,
            'original_length': len(original) if original else 0,
            'summary_length': len(summary) if summary else 0,
            'metrics': metrics,
            'summary_preview': (summary[:200] + '...') if summary and len(summary) > 200 else (summary if summary else "")
        }
        
        self.metrics_history.append(evaluation)
        return evaluation
    
    def print_detailed_analysis(self, evaluations):
        """Análisis detallado con insights."""
        print("\n" + "="*100)
        print("ANÁLISIS COMPREHENSIVO DE CALIDAD DE RESUMEN")
        print("="*100)
        
        if not evaluations:
            print("❌ No hay evaluaciones para analizar.")
            return
        
        for eval_data in evaluations:
            print(f"\n📊 MÉTODO: {eval_data['method']}")
            print(f"   📝 Resumen: {eval_data.get('summary_preview', 'N/A')}")
            print(f"   📐 Longitud: {eval_data.get('summary_length', 0)} carac. (Compresión: {eval_data.get('compression_ratio', 0):.1%})")
            
            metrics = eval_data.get('metrics', {})
            print(f"   🎯 PUNTAJE GENERAL: {metrics.get('overall_score', 0):.4f}")
            print(f"   📈 Métricas detalladas:")
            print(f"      • ROUGE-like: {metrics.get('rouge_like_score', 0):.4f} (cobertura de contenido)")
            print(f"      • BLEU: {metrics.get('bleu_score', 0):.4f} (similitud lexical)")
            print(f"      • Cobertura: {metrics.get('coverage_score', 0):.4f} (frases clave cubiertas)")
            print(f"      • Coherencia: {metrics.get('coherence_score', 0):.4f} (fluidez del resumen)")
            print(f"      • Diversidad: {metrics.get('diversity_score', 0):.4f} (variedad lexical)")
            print(f"      • Redundancia: {metrics.get('redundancy_score', 0):.4f} (repetición)")
        
        # Análisis comparativo
        try:
            best_overall = max(evaluations, key=lambda x: x.get('metrics', {}).get('overall_score', 0))
            best_rouge = max(evaluations, key=lambda x: x.get('metrics', {}).get('rouge_like_score', 0))
            best_coverage = max(evaluations, key=lambda x: x.get('metrics', {}).get('coverage_score', 0))
            best_coherence = max(evaluations, key=lambda x: x.get('metrics', {}).get('coherence_score', 0))
            
            print(f"\n🏆 RECOMENDACIONES:")
            print(f"   • MEJOR GENERAL: {best_overall['method']} (Score: {best_overall['metrics']['overall_score']:.4f})")
            print(f"   • MEJOR COBERTURA: {best_coverage['method']} (Cobertura: {best_coverage['metrics']['coverage_score']:.4f})")
            print(f"   • MEJOR ROUGE: {best_rouge['method']} (ROUGE: {best_rouge['metrics']['rouge_like_score']:.4f})")
            print(f"   • MEJOR COHERENCIA: {best_coherence['method']} (Coherencia: {best_coherence['metrics']['coherence_score']:.4f})")
            
        except Exception as e:
            print(f"⚠️  Error en análisis comparativo: {e}")

    def get_best_method(self, evaluations):
        """Obtiene el mejor método basado en el score general."""
        if not evaluations:
            return None
            
        try:
            return max(evaluations, key=lambda x: x.get('metrics', {}).get('overall_score', 0))
        except:
            return None

    def export_metrics_to_csv(self, filename="metrics_report.csv"):
        """Exporta las métricas a un archivo CSV."""
        try:
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['method', 'compression_ratio', 'original_length', 'summary_length', 
                             'overall_score', 'rouge_like_score', 'bleu_score', 'coverage_score',
                             'coherence_score', 'diversity_score', 'redundancy_score']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for eval_data in self.metrics_history:
                    metrics = eval_data.get('metrics', {})
                    row = {
                        'method': eval_data.get('method', ''),
                        'compression_ratio': eval_data.get('compression_ratio', 0),
                        'original_length': eval_data.get('original_length', 0),
                        'summary_length': eval_data.get('summary_length', 0),
                        'overall_score': metrics.get('overall_score', 0),
                        'rouge_like_score': metrics.get('rouge_like_score', 0),
                        'bleu_score': metrics.get('bleu_score', 0),
                        'coverage_score': metrics.get('coverage_score', 0),
                        'coherence_score': metrics.get('coherence_score', 0),
                        'diversity_score': metrics.get('diversity_score', 0),
                        'redundancy_score': metrics.get('redundancy_score', 0)
                    }
                    writer.writerow(row)
                    
            print(f"✅ Métricas exportadas a {filename}")
            
        except Exception as e:
            print(f"❌ Error exportando métricas: {e}")

# Función de utilidad para evaluación rápida
def quick_evaluate(original, summary):
    """
    Evaluación rápida sin necesidad de processed_data.
    
    Args:
        original (str): Texto original
        summary (str): Resumen generado
        
    Returns:
        dict: Métricas básicas
    """
    evaluator = MetricsEvaluator()
    return evaluator.calculate_all_metrics(original, summary)