"""
Pipeline de Resumen Extractivo Avanzado
=======================================

Un sistema modular de resumen extractivo multilingüe que combina TF-ICF 
con clustering semántico y métricas avanzadas de evaluación.
"""

from .text_preprocessor import EnhancedTextPreprocessor
from .semantic_summarizer import SemanticTFICFSummarizer
from .metrics_evaluator import MetricsEvaluator, AdvancedSummaryEvaluator

__version__ = "1.0.0"
__author__ = "Daryll Lorenzo Alfonso"
__all__ = [
    "EnhancedTextPreprocessor",
    "SemanticTFICFSummarizer", 
    "MetricsEvaluator",
    "AdvancedSummaryEvaluator",
    "summarization_pipeline"
]

# Pipeline por defecto
from sklearn.pipeline import Pipeline
summarization_pipeline = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor()),
    ('summarizer', SemanticTFICFSummarizer(n_sentences='auto', clustering_method='kmeans'))
])