import time
import logging
from typing import List, Dict, Any, Optional
from sklearn.pipeline import Pipeline

from text_preprocessor import EnhancedTextPreprocessor
from semantic_summarizer import SemanticTFICFSummarizer
from metrics_evaluator import AdvancedSummaryEvaluator

# Configurar logging
logger = logging.getLogger(__name__)

class SummaryService:
    """Servicio para gestión de resúmenes con lógica de negocio"""
    
    def __init__(self):
        self.pipeline = self._initialize_pipeline()
        self.evaluator = AdvancedSummaryEvaluator()
        self._service_ready = True
        self._startup_time = time.time()
    
    def _initialize_pipeline(self) -> Pipeline:
        """Inicializar y configurar el pipeline de procesamiento"""
        return Pipeline([
            ('preprocessor', EnhancedTextPreprocessor()),
            ('summarizer', SemanticTFICFSummarizer(
                n_sentences='auto', 
                clustering_method='kmeans'
            ))
        ])
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar estado del servicio"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "model_ready": self._service_ready,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def generate_summary(
        self, 
        text: str, 
        n_sentences: Optional[int] = None,
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Generar resumen de un texto con métricas opcionales
        
        Args:
            text: Texto a resumir
            n_sentences: Número de oraciones en el resumen
            include_metrics: Incluir métricas de evaluación
            
        Returns:
            Dict con resultados del resumen
        """
        start_time = time.time()
        
        # Validaciones de negocio
        self._validate_text_input(text)
        
        # Configurar pipeline según parámetros
        if n_sentences is not None:
            self.pipeline.named_steps['summarizer'].n_sentences = n_sentences
        
        # Procesar texto
        results = self.pipeline.fit_transform([text])
        
        if not results:
            raise ValueError("No se pudo generar el resumen")
        
        result = results[0]
        processing_time = time.time() - start_time
        
        # Asegurarnos de que todos los campos requeridos estén presentes
        response_data = {
            "summary": result.get('summary', ''),
            "original_length": result.get('original_length', len(text)),
            "summary_length": result.get('summary_length', len(result.get('summary', ''))),
            "compression_ratio": result.get('compression_ratio', 0.0),
            "language": result.get('language', 'unknown'),
            "selected_sentences": result.get('selected_sentences', []),
            "processing_time": processing_time,
            "key_phrases": [
                phrase for phrase, score in result.get('key_phrases', [])[:10]
            ] if result.get('key_phrases') else []
        }
        
        # Calcular métricas si se solicitan
        if include_metrics:
            response_data["metrics"] = self._calculate_metrics(text, result)
        
        logger.info(
            f"Resumen generado: {response_data['compression_ratio']:.1%} compresión, "
            f"{processing_time:.2f}s, {len(response_data['selected_sentences'])} oraciones"
        )
        
        return response_data
    
    def generate_batch_summaries(
        self, 
        texts: List[str], 
        n_sentences: Optional[int] = None,
        include_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Generar resúmenes para múltiples textos
        
        Args:
            texts: Lista de textos a resumir
            n_sentences: Número de oraciones por resumen
            include_metrics: Incluir métricas para cada texto
            
        Returns:
            Dict con resultados del procesamiento por lotes
        """
        start_time = time.time()
        summaries = []
        failed_texts = []
        
        # Configurar número de oraciones fijo para batch
        if n_sentences:
            self.pipeline.named_steps['summarizer'].n_sentences = n_sentences
        
        for i, text in enumerate(texts):
            try:
                # Validar texto mínimo para batch
                if len(text.strip()) < 50:
                    logger.warning(f"Texto {i} muy corto, omitiendo")
                    failed_texts.append({"index": i, "reason": "Texto muy corto"})
                    continue
                
                # Generar resumen individual
                summary_result = self.generate_summary(
                    text, 
                    n_sentences=n_sentences, 
                    include_metrics=include_metrics
                )
                
                summaries.append(summary_result)
                
            except Exception as e:
                logger.error(f"Error procesando texto {i}: {e}")
                failed_texts.append({"index": i, "reason": str(e)})
                continue
        
        total_time = time.time() - start_time
        
        # Calcular estadísticas
        stats = self._calculate_batch_statistics(summaries)
        
        return {
            "summaries": summaries,
            "total_processed": len(summaries),
            "failed_count": len(failed_texts),
            "failed_texts": failed_texts,
            "statistics": stats,
            "total_processing_time": total_time
        }
    
    def compare_summaries(
        self, 
        original: str, 
        summary: str
    ) -> Dict[str, Any]:
        """
        Comparar un texto original con un resumen usando métricas
        
        Args:
            original: Texto original
            summary: Texto resumen a comparar
            
        Returns:
            Dict con resultados de la comparación
        """
        # Validaciones
        if len(original.strip()) < 100:
            raise ValueError("El texto original debe tener al menos 100 caracteres")
        
        if len(summary.strip()) < 20:
            raise ValueError("El resumen debe tener al menos 20 caracteres")
        
        # Procesar el texto original para obtener frases clave
        preprocessor = EnhancedTextPreprocessor()
        processed_data = preprocessor.fit_transform([original])
        
        if not processed_data:
            raise ValueError("No se pudo procesar el texto original")
        
        processed_info = processed_data[0]
        
        # Evaluar el resumen proporcionado
        evaluation = self.evaluator.comprehensive_evaluation(
            original,
            summary,
            "Comparación Manual",
            processed_info,
            list(range(len(summary.split('.'))))  # Estimación simple
        )
        
        return {
            "comparison_metrics": evaluation['metrics'],
            "details": {
                "original_length": len(original),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(original),
                "key_phrases_coverage": evaluation.get('key_phrases_coverage', 0),
                "coherence_score": evaluation['metrics'].get('coherence', 0)
            },
            "quality_assessment": self._assess_quality(evaluation['metrics'])
        }
    
    def _validate_text_input(self, text: str) -> None:
        """Validar texto de entrada según reglas de negocio"""
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")
        
        if len(text.strip()) < 100:
            raise ValueError("El texto debe tener al menos 100 caracteres")
        
        # Validar que tenga suficiente contenido (más de 3 oraciones)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len([s for s in sentences if len(s) > 20]) < 2:
            raise ValueError("El texto debe contener al menos 2 oraciones significativas")
    
    def _calculate_metrics(
        self, 
        original_text: str, 
        result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcular métricas de evaluación para un resumen"""
        try:
            processed_data_for_metrics = {
                'key_phrases': result.get('key_phrases', []),
                'sentences': result.get('sentences', []),
                'original': result.get('original', original_text)
            }
            
            evaluation = self.evaluator.comprehensive_evaluation(
                original_text,
                result.get('summary', ''),
                "API Summary",
                processed_data_for_metrics,
                result.get('selected_sentences', [])
            )
            
            return evaluation['metrics']
            
        except Exception as e:
            logger.warning(f"Error calculando métricas: {e}")
            return {
                "rouge_like_score": 0.0,
                "bleu_score": 0.0,
                "coherence": 0.0,
                "overall_score": 0.0
            }
    
    def _calculate_batch_statistics(
        self, 
        summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calcular estadísticas para procesamiento por lotes"""
        if not summaries:
            return {
                "average_compression": 0.0,
                "min_compression": 0.0,
                "max_compression": 0.0,
                "average_summary_length": 0.0,
                "language_distribution": {},
                "total_summaries": 0
            }
        
        compression_ratios = [s.get('compression_ratio', 0.0) for s in summaries]
        summary_lengths = [s.get('summary_length', 0) for s in summaries]
        
        # Detectar idiomas
        languages = {}
        for summary in summaries:
            lang = summary.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            "average_compression": sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0,
            "min_compression": min(compression_ratios) if compression_ratios else 0.0,
            "max_compression": max(compression_ratios) if compression_ratios else 0.0,
            "average_summary_length": sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0.0,
            "language_distribution": languages,
            "total_summaries": len(summaries)
        }
    
    def _assess_quality(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluar la calidad del resumen basado en métricas"""
        rouge_score = metrics.get('rouge_like_score', 0)
        bleu_score = metrics.get('bleu_score', 0)
        coherence = metrics.get('coherence', 0)
        overall = metrics.get('overall_score', 0)
        
        # Lógica de evaluación de calidad
        quality = "BAJA"
        if overall >= 0.7:
            quality = "EXCELENTE"
        elif overall >= 0.5:
            quality = "BUENA"
        elif overall >= 0.3:
            quality = "ACEPTABLE"
        
        issues = []
        if rouge_score < 0.3:
            issues.append("Baja cobertura de contenido")
        if bleu_score < 0.2:
            issues.append("Baja similitud léxica")
        if coherence < 0.5:
            issues.append("Problemas de coherencia")
        
        return {
            "quality_level": quality,
            "overall_score": overall,
            "issues": issues,
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generar recomendaciones para mejorar el resumen"""
        recommendations = []
        
        if metrics.get('rouge_like_score', 0) < 0.4:
            recommendations.append("Considerar incluir más oraciones clave del texto original")
        
        if metrics.get('coherence', 0) < 0.6:
            recommendations.append("Revisar la fluidez entre oraciones del resumen")
        
        return recommendations

    def update_pipeline_parameters(
        self, 
        n_sentences: Optional[int] = None,
        min_word_length: Optional[int] = None,
        diversity_weight: Optional[float] = None
    ) -> Dict[str, Any]:
        """Actualizar parámetros del pipeline en tiempo de ejecución"""
        updates = {}
        
        if n_sentences is not None:
            self.pipeline.named_steps['summarizer'].n_sentences = n_sentences
            updates['n_sentences'] = n_sentences
        
        if min_word_length is not None:
            self.pipeline.named_steps['preprocessor'].min_word_length = min_word_length
            updates['min_word_length'] = min_word_length
        
        if diversity_weight is not None:
            # Nota: diversity_weight podría no existir en tu summarizer actual
            # Si no existe, puedes omitir esta actualización o agregar el atributo
            if hasattr(self.pipeline.named_steps['summarizer'], 'diversity_weight'):
                self.pipeline.named_steps['summarizer'].diversity_weight = diversity_weight
                updates['diversity_weight'] = diversity_weight
        
        logger.info(f"Parámetros actualizados: {updates}")
        return {"updated_parameters": updates, "status": "success"}
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de uso del servicio"""
        uptime = time.time() - self._startup_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        
        return {
            "service_uptime": f"{hours}h {minutes}m",
            "total_summaries_generated": "N/A",  # Podrías implementar contador
            "average_processing_time": "N/A",
            "memory_usage": "N/A"
        }

# Instancia global del servicio
summary_service = SummaryService()