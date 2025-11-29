from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class QualityLevel(str, Enum):
    """Niveles de calidad para evaluación de resúmenes"""
    EXCELLENT = "EXCELENTE"
    GOOD = "BUENA"
    ACCEPTABLE = "ACEPTABLE"
    LOW = "BAJA"

class SummaryRequest(BaseModel):
    """Modelo para solicitud de resumen"""
    text: str = Field(
        ..., 
        description="Texto a resumir", 
        min_length=100,
        example="La inteligencia artificial está transformando radicalmente el panorama tecnológico global. Los avances en machine learning y deep learning han permitido desarrollar sistemas capaces de realizar tareas que antes se consideraban exclusivamente humanas."
    )
    n_sentences: Optional[int] = Field(
        None, 
        description="Número de oraciones en el resumen (auto para cálculo automático)",
        example=3
    )
    language: Optional[str] = Field(
        None, 
        description="Idioma del texto (auto para detección automática)",
        example="auto"
    )
    include_metrics: Optional[bool] = Field(
        True, 
        description="Incluir métricas de evaluación",
        example=True
    )

class SummaryResponse(BaseModel):
    """Modelo para respuesta del resumen"""
    summary: str = Field(..., description="Texto resumido")
    original_length: int = Field(..., description="Longitud del texto original")
    summary_length: int = Field(..., description="Longitud del resumen")
    compression_ratio: float = Field(..., description="Ratio de compresión")
    language: str = Field(..., description="Idioma detectado")
    selected_sentences: List[int] = Field(..., description="Índices de oraciones seleccionadas")
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    
    # Métricas opcionales
    metrics: Optional[Dict[str, float]] = Field(None, description="Métricas de evaluación")
    key_phrases: Optional[List[str]] = Field(None, description="Frases clave identificadas")

    class Config:
        json_schema_extra = {
            "example": {
                "summary": "La inteligencia artificial transforma el panorama tecnológico. Los avances en machine learning permiten sistemas capaces de tareas humanas.",
                "original_length": 250,
                "summary_length": 80,
                "compression_ratio": 0.32,
                "language": "es",
                "selected_sentences": [0, 2],
                "processing_time": 1.5,
                "metrics": {
                    "rouge_like_score": 0.65,
                    "bleu_score": 0.45,
                    "coherence": 0.78,
                    "overall_score": 0.72
                },
                "key_phrases": ["inteligencia artificial", "machine learning", "deep learning"]
            }
        }

class BatchSummaryRequest(BaseModel):
    """Modelo para procesamiento por lotes"""
    texts: List[str] = Field(
        ..., 
        description="Lista de textos a resumir", 
        min_items=1,
        example=[
            "Primer texto largo para resumir...",
            "Segundo texto largo para resumir..."
        ]
    )
    n_sentences: Optional[int] = Field(
        3, 
        description="Número de oraciones por resumen",
        example=3
    )
    include_metrics: Optional[bool] = Field(
        False, 
        description="Incluir métricas para cada texto",
        example=False
    )

class BatchSummaryResponse(BaseModel):
    """Respuesta para procesamiento por lotes"""
    summaries: List[SummaryResponse] = Field(..., description="Lista de resúmenes generados")
    total_processed: int = Field(..., description="Total de textos procesados")
    average_compression: float = Field(..., description="Compresión promedio")
    total_processing_time: float = Field(..., description="Tiempo total de procesamiento")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión de la API")
    model_ready: bool = Field(..., description="Si el modelo está cargado y listo")
    timestamp: str = Field(..., description="Timestamp de la respuesta")

    model_config = {
        "protected_namespaces": ()
    }

class ErrorResponse(BaseModel):
    """Modelo para respuestas de error"""
    error: str = Field(..., description="Mensaje de error")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalles adicionales del error")
    code: int = Field(..., description="Código de error HTTP")

class ServiceMetricsResponse(BaseModel):
    """Modelo para métricas del servicio"""
    service_uptime: str = Field(..., description="Tiempo de actividad del servicio")
    total_summaries_generated: str = Field(..., description="Total de resúmenes generados")
    average_processing_time: str = Field(..., description="Tiempo promedio de procesamiento")
    memory_usage: str = Field(..., description="Uso de memoria del servicio")

class ConfigUpdateResponse(BaseModel):
    """Modelo para respuesta de actualización de configuración"""
    updated_parameters: Dict[str, Any] = Field(..., description="Parámetros actualizados")
    status: str = Field(..., description="Estado de la operación")

class ComparisonResponse(BaseModel):
    """Modelo para respuesta de comparación de resúmenes"""
    comparison_metrics: Dict[str, float] = Field(..., description="Métricas de comparación")
    details: Dict[str, Any] = Field(..., description="Detalles de la comparación")
    quality_assessment: Dict[str, Any] = Field(..., description="Evaluación de calidad")