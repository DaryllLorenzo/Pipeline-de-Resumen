# 📝 **README Actualizado**


# Pipeline de Resumen Extractivo Avanzado

Un sistema de resumen extractivo multilingüe implementado como pipeline modular de scikit-learn que combina algoritmos TF-ICF mejorados con clustering semántico para generar resúmenes de alta calidad.

## 🚀 Características Mejoradas

- **🔍 Resumen extractivo semántico** - Combina TF-ICF con análisis de frases clave
- **🌍 Soporte multilingüe inteligente** - Español e inglés con detección avanzada
- **🎯 Algoritmo TF-ICF mejorado** - Con suavizado y ponderación de términos
- **📊 Clustering semántico** - Para diversidad temática en los resúmenes
- **⚡ Pipeline modular** - Arquitectura separada en componentes reutilizables
- **📈 Métricas avanzadas** - Evaluación integral con BLEU, ROUGE, coherencia y más
- **🛡️ Manejo robusto de errores** - Fallbacks inteligentes para todos los casos edge
- **💾 Mínimas dependencias** - Solo scikit-learn, numpy y nltk básico

## 🏗️ Arquitectura Modular

```
summarization_pipeline/
├── 📁 text_preprocessor.py    # Procesamiento y limpieza de texto
├── 📁 semantic_summarizer.py  # Algoritmo principal de resumen
├── 📁 metrics_evaluator.py    # Evaluación y métricas de calidad
├── 📁 main.py                 # Ejemplos y uso principal
└── 📁 __init__.py            # Configuración del paquete
```

## 📋 Requisitos Mejorados

```bash
pip install scikit-learn numpy nltk
```

## 🧠 Algoritmos Avanzados Implementados

### TF-ICF Mejorado
- **Suavizado de Laplace** para evitar divisiones por cero
- **Ponderación de términos** por longitud e informatividad
- **ICF balanceado** que no castiga demasiado términos comunes

### Clustering Semántico
- **K-means adaptativo** basado en longitud del texto
- **Agrupamiento por similitud** como fallback robusto
- **Selección por clusters** para diversidad temática

### Scores Multi-dimensionales
```python
combined_score = (
    tf_icf * 0.35 +        # TF-ICF tradicional mejorado
    key_phrase * 0.25 +     # Frases clave del documento
    semantic * 0.15 +       # Análisis semántico del preprocesador
    position * 0.15 +       # Posición en el texto (curva U)
    length * 0.10           # Longitud óptima de oraciones
)
```

## 🛠️ Uso Básico Mejorado

### Ejemplo Simple

```python
from summarization_pipeline import summarization_pipeline

# Texto a resumir
texto = """
La inteligencia artificial está transformando radicalmente el panorama tecnológico global. 
Los avances en machine learning y deep learning han permitido desarrollar sistemas capaces 
de realizar tareas que antes se consideraban exclusivamente humanas. En el campo de la medicina, 
los algoritmos de IA pueden analizar imágenes médicas con una precisión que rivaliza con 
la de radiólogos expertos. Esto ha llevado a diagnósticos más tempranos y precisos de 
enfermedades como el cáncer, mejorando significativamente las tasas de supervivencia.
"""

# Procesar y obtener resumen
resultados = summarization_pipeline.fit_transform([texto])
resumen = resultados[0]['summary']
metricas = resultados[0]['metrics']  # Nuevo: métricas incluidas

print("Resumen:", resumen)
print("Compresión:", f"{resultados[0]['compression_ratio']:.1%}")
print("Score General:", f"{metricas['overall_score']:.4f}")
```

### Evaluación Avanzada de Calidad

```python
from metrics_evaluator import AdvancedSummaryEvaluator

evaluator = AdvancedSummaryEvaluator()
evaluacion = evaluator.comprehensive_evaluation(
    texto_original, 
    resumen, 
    "Mi Método",
    processed_data=resultado,           # Datos para métricas avanzadas
    selected_indices=resultado['selected_sentences']
)

print("Métricas detalladas:")
print(f"• ROUGE-like: {evaluacion['metrics']['rouge_like_score']:.4f}")
print(f"• BLEU: {evaluacion['metrics']['bleu_score']:.4f}")
print(f"• Coherencia: {evaluacion['metrics']['coherence_score']:.4f}")
print(f"• Cobertura: {evaluacion['metrics']['coverage_score']:.4f}")
```

## 📊 Métricas de Evaluación Implementadas

| Métrica | Descripción | Rango Óptimo |
|---------|-------------|--------------|
| **ROUGE-like** | Cobertura de contenido vs original | 0.4-0.7 |
| **BLEU Score** | Similitud lexical con referencias | 0.3-0.6 |
| **Coherencia** | Fluidez entre oraciones del resumen | 0.6-1.0 |
| **Cobertura** | Frases clave del original incluidas | 0.7-1.0 |
| **Diversidad** | Variedad lexical en el resumen | 0.7-0.9 |
| **Redundancia** | Nivel de repetición (menos es mejor) | 0.0-0.2 |

## ⚙️ Personalización Avanzada

### Pipeline con Configuración Específica

```python
from sklearn.pipeline import Pipeline
from text_preprocessor import EnhancedTextPreprocessor
from semantic_summarizer import SemanticTFICFSummarizer

# Pipeline personalizado para documentos largos
pipeline_largo = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor(min_word_length=3)),
    ('summarizer', SemanticTFICFSummarizer(
        n_sentences='auto',           # Cálculo automático
        clustering_method='kmeans',   # Clustering semántico
        diversity_weight=0.4          # Énfasis en diversidad
    ))
])
```

### Dominios Específicos con Bonus Temático

```python
class MedicalSummarizer(SemanticTFICFSummarizer):
    def __init__(self, n_sentences='auto'):
        super().__init__(n_sentences)
        self.medical_terms = {
            'diagnóstico', 'tratamiento', 'síntomas', 'paciente', 
            'enfermedad', 'medicamento', 'hospital', 'cáncer'
        }
    
    def calculate_semantic_scores(self, processed_data):
        scores = super().calculate_semantic_scores(processed_data)
        
        # Bonus para términos médicos
        for i, (idx, score, length) in enumerate(scores):
            sentence = processed_data['sentences'][idx].lower()
            medical_bonus = sum(1 for term in self.medical_terms if term in sentence)
            medical_bonus = min(medical_bonus * 0.1, 0.3)  # Máximo 30% bonus
            scores[i] = (idx, score * (1 + medical_bonus), length)
        
        return scores
```

## 📈 Métodos de Evaluación

### Evaluación Automática
```python
# Evaluación completa con todos los componentes
results = pipeline.fit_transform([texto_largo])
evaluation = evaluator.comprehensive_evaluation(
    texto_largo, 
    results[0]['summary'], 
    "Enhanced TF-ICF",
    results[0],
    results[0]['selected_sentences']
)

# Exportar resultados
evaluator.export_metrics_to_csv("evaluacion_completa.csv")
```

### Comparación de Métodos
```python
methods = {
    "Básico": basic_pipeline,
    "Con Clustering": clustered_pipeline, 
    "Avanzado": advanced_pipeline
}

for name, pipeline in methods.items():
    results = pipeline.fit_transform([texto])
    # Evaluar y comparar...
```

## 🚀 Rendimiento y Optimización

- **⚡ Procesamiento eficiente**: Solo CPU, sin modelos grandes
- **📐 Escalabilidad**: Maneja documentos de 100 a 10,000 palabras
- **🔄 Cache opcional**: Para procesamiento repetitivo
- **🎯 Balance calidad/velocidad**: Optimizado para uso práctico

## 🔮 Próximas Mejoras

- [ ] Soporte para más idiomas (francés, portugués, alemán)
- [ ] Integración con modelos de embeddings livianos
- [ ] Interfaz web con Streamlit o FastAPI
- [ ] Análisis de sentimiento en resúmenes
- [ ] Optimización para dominios específicos (legal, médico, técnico)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.
