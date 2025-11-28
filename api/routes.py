from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging

from .models import (
    SummaryRequest, SummaryResponse, BatchSummaryRequest,
    BatchSummaryResponse, HealthResponse, ErrorResponse,
    ServiceMetricsResponse, ConfigUpdateResponse,
    ComparisonResponse
)
from .services import summary_service

# Configurar logging
logger = logging.getLogger(__name__)

# Crear router
router = APIRouter(prefix="/api/v1", tags=["summarization"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de salud del servicio"""
    try:
        health_info = summary_service.health_check()
        return HealthResponse(**health_info)
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        raise HTTPException(status_code=500, detail="Service health check failed")

@router.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    """
    Generar resumen de un texto individual
    
    - **text**: Texto a resumir (mínimo 100 caracteres)
    - **n_sentences**: Número de oraciones (opcional, 'auto' por defecto)
    - **language**: Idioma (opcional, detección automática)
    - **include_metrics**: Incluir métricas de evaluación
    """
    try:
        # Delegar toda la lógica al servicio
        result = summary_service.generate_summary(
            text=request.text,
            n_sentences=request.n_sentences,
            include_metrics=request.include_metrics
        )
        
        return SummaryResponse(**result)
        
    except ValueError as e:
        logger.warning(f"Validación fallida: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en summarize_text: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

@router.post("/summarize/batch", response_model=BatchSummaryResponse)
async def summarize_batch(request: BatchSummaryRequest):
    """
    Generar resúmenes para múltiples textos
    
    - **texts**: Lista de textos a resumir
    - **n_sentences**: Número de oraciones por resumen
    - **include_metrics**: Incluir métricas para cada texto
    """
    try:
        result = summary_service.generate_batch_summaries(
            texts=request.texts,
            n_sentences=request.n_sentences,
            include_metrics=request.include_metrics
        )
        
        # Convertir los resúmenes al modelo Response
        summary_responses = []
        for summary in result["summaries"]:
            summary_responses.append(SummaryResponse(**summary))
        
        return BatchSummaryResponse(
            summaries=summary_responses,
            total_processed=result["total_processed"],
            average_compression=result["statistics"]["average_compression"],
            total_processing_time=result["total_processing_time"]
        )
        
    except Exception as e:
        logger.error(f"Error en summarize_batch: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando lote: {str(e)}"
        )

@router.get("/metrics/compare", response_model=ComparisonResponse)
async def compare_summaries(original: str, summary: str):
    """
    Comparar un texto original con un resumen usando métricas
    
    - **original**: Texto original
    - **summary**: Texto resumen a comparar
    """
    try:
        result = summary_service.compare_summaries(original, summary)
        return ComparisonResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en compare_summaries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en comparación: {str(e)}"
        )

@router.put("/config", response_model=ConfigUpdateResponse)
async def update_config(
    n_sentences: int = None,
    min_word_length: int = None,
    diversity_weight: float = None
):
    """
    Actualizar configuración del pipeline en tiempo de ejecución
    
    - **n_sentences**: Número de oraciones para resúmenes
    - **min_word_length**: Longitud mínima de palabras
    - **diversity_weight**: Peso para diversidad en scoring
    """
    try:
        result = summary_service.update_pipeline_parameters(
            n_sentences=n_sentences,
            min_word_length=min_word_length,
            diversity_weight=diversity_weight
        )
        return ConfigUpdateResponse(**result)
        
    except Exception as e:
        logger.error(f"Error actualizando configuración: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error actualizando configuración: {str(e)}"
        )

@router.get("/metrics/service", response_model=ServiceMetricsResponse)
async def get_service_metrics():
    """Obtener métricas de uso del servicio"""
    try:
        metrics = summary_service.get_service_metrics()
        return ServiceMetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"Error obteniendo métricas del servicio: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo métricas: {str(e)}"
        )