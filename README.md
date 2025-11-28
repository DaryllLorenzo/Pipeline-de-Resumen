# Pipeline de Resumen Extractivo Avanzado

Un sistema modular de resumen extractivo multilingüe que implementa algoritmos avanzados de procesamiento de lenguaje natural usando scikit-learn. Diseñado para ser eficiente, extensible y con dependencias mínimas.

## 🏗️ **Arquitectura del Sistema**

### **Módulos Principales y su Funcionamiento**

```
summarization_pipeline/
├── 📁 text_preprocessor.py    # Procesamiento y limpieza de texto
├── 📁 semantic_summarizer.py  # Algoritmo principal de resumen
├── 📁 metrics_evaluator.py    # Evaluación y métricas de calidad
├── 📁 main.py                 # Ejemplos y uso principal
└── 📁 __init__.py            # Configuración del paquete
```

#### **1. 📁 text_preprocessor.py - Procesamiento Inteligente de Texto**

**Propósito**: Preparar y limpiar el texto para el análisis semántico.

**Flujo de procesamiento**:
```
texto_entrante → detect_language() → split_sentences() → preprocess_text() → datos_estructurados
```

**Técnicas Implementadas**:

- **🔤 Detección de Idioma Mejorada**:
  Combina múltiples heurísticas: caracteres especiales (á, é, í, ó, ú, ñ), palabras comunes por idioma, y longitud promedio de palabras para determinar si el texto es español o inglés.

- **📝 División de Oraciones con NLTK**:
  Utiliza tokenización inteligente de NLTK para dividir el texto en oraciones, filtrando aquellas muy cortas (<20 caracteres) que suelen contener poca información.

- **🔍 Extracción de Frases Clave**:
  Identifica n-gramas importantes (1-3 palabras) usando TF-IDF. Ejemplo: `[("aprendizaje automático", 0.85), ("inteligencia artificial", 0.78)]`

- **🧹 Preprocesamiento de Texto**:
  Convierte a minúsculas, elimina puntuación, remueve stopwords y filtra palabras muy cortas para limpiar el texto manteniendo el contexto semántico.

**Salida**: Diccionario estructurado con metadatos del texto procesado, incluyendo oraciones originales, oraciones preprocesadas, frases clave y puntuaciones semánticas.

---

#### **2. 📁 semantic_summarizer.py - Algoritmo Principal de Resumen**

**Propósito**: Seleccionar las oraciones más importantes usando TF-ICF mejorado y clustering semántico.

**Técnicas Implementadas**:

- **🎯 TF-ICF Mejorado (Term Frequency - Inverse Class Frequency)**:
  ```python
  # Fórmula mejorada:
  TF(término) = (frecuencia en oración) / (total términos) * peso_longitud
  ICF(término) = log(total_oraciones / docs_con_término) + ajuste
  Score = Σ [TF(t) × ICF(t)] para cada término t
  ```
  El TF-ICF identifica términos que son importantes dentro de una oración pero poco comunes en otras oraciones del mismo texto.

- **📊 Sistema de Scoring Multi-dimensional**:
  Combina múltiples factores con pesos optimizados:
  - 35% TF-ICF (relevancia léxica)
  - 30% Frases clave del documento
  - 20% Análisis semántico del preprocesador
  - 15% Posición estratégica (curva en U)

- **🎪 Clustering Semántico Adaptativo**:
  Agrupa oraciones similares temáticamente usando K-means sobre representaciones TF-IDF, asegurando diversidad en el resumen final.

- **🔄 Estrategia de Selección en 3 Fases**:
  1. **Mejor por cluster** - Garantiza diversidad temática
  2. **Segunda mejor de clusters grandes** - Añade profundidad
  3. **Mejores globales restantes** - Asegura máxima relevancia

**Salida**: Resumen estructurado con métricas de compresión, oraciones seleccionadas y datos para evaluación.

---

#### **3. 📁 metrics_evaluator.py - Evaluación de Calidad**

**Propósito**: Medir la calidad del resumen usando métricas estandarizadas.

**Métricas Implementadas**:

- **📈 ROUGE-like Score**: Mide cobertura de contenido comparando la superposición de palabras entre el resumen y el original.

- **🔤 BLEU Score Mejorado**: Evalúa similitud n-gram con referencias múltiples usando smoothing para evitar zeros.

- **🔄 Coherencia**: Calcula la fluidez entre oraciones del resumen usando similitud de coseno entre representaciones vectoriales consecutivas.

- **🎯 Cobertura Semántica**: Porcentaje de frases clave del documento original que están incluidas en el resumen.

- **📊 Score General Ponderado**: Combina todas las métricas con pesos optimizados para un evaluación integral.

---

## 🚀 **Uso del Pipeline en tus Programas Python**

### **1. Uso Básico - Resumen Simple**

```python
from sklearn.pipeline import Pipeline
from text_preprocessor import EnhancedTextPreprocessor
from semantic_summarizer import SemanticTFICFSummarizer

# Crear pipeline
pipeline = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor()),
    ('summarizer', SemanticTFICFSummarizer(n_sentences=3))
])

# Texto a resumir
texto_largo = "Tu texto largo aquí..."

# Generar resumen
resultados = pipeline.fit_transform([texto_largo])
resumen = resultados[0]['summary']

print(f"📝 Resumen: {resumen}")
print(f"📊 Compresión: {resultados[0]['compression_ratio']:.1%}")
print(f"🔤 Idioma: {resultados[0]['language']}")
```

### **2. Uso Avanzado - Con Evaluación de Calidad**

```python
from metrics_evaluator import AdvancedSummaryEvaluator

# Pipeline con configuración avanzada
pipeline_avanzado = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor(
        min_word_length=3,
        use_semantic_analysis=True
    )),
    ('summarizer', SemanticTFICFSummarizer(
        n_sentences='auto',           # Cálculo automático
        clustering_method='kmeans',   # Clustering semántico
        diversity_weight=0.4          # Énfasis en diversidad
    ))
])

# Procesar y evaluar
resultados = pipeline_avanzado.fit_transform([texto_largo])
evaluator = AdvancedSummaryEvaluator()

evaluacion = evaluator.comprehensive_evaluation(
    texto_largo,
    resultados[0]['summary'],
    "Mi Resumen",
    resultados[0],
    resultados[0]['selected_sentences']
)

print(f"🎯 Score General: {evaluacion['metrics']['overall_score']:.3f}")
print(f"📈 ROUGE: {evaluacion['metrics']['rouge_like_score']:.3f}")
print(f"🔤 BLEU: {evaluacion['metrics']['bleu_score']:.3f}")
```

### **3. Personalización para Dominios Específicos**

```python
class MedicalSummarizer(SemanticTFICFSummarizer):
    """Summarizer especializado en textos médicos"""
    
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
            medical_bonus = min(medical_bonus * 0.15, 0.3)
            scores[i] = (idx, score * (1 + medical_bonus), length)
        
        return scores

# Pipeline médico especializado
pipeline_medico = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor()),
    ('summarizer', MedicalSummarizer(n_sentences=4))
])
```

### **4. Procesamiento por Lotes**

```python
import pandas as pd

# Procesar múltiples documentos
documentos = [texto1, texto2, texto3, texto4]
resultados = pipeline.fit_transform(documentos)

# Crear DataFrame con resultados
df_resultados = pd.DataFrame([{
    'resumen': r['summary'],
    'compresion': r['compression_ratio'],
    'idioma': r['language'],
    'oraciones_seleccionadas': len(r['selected_sentences'])
} for r in resultados])

df_resultados.to_csv('resumenes_generados.csv', index=False)
```

## 📊 **Métricas de Evaluación**

| Métrica | Descripción | Rango Óptimo | Interpretación |
|---------|-------------|--------------|----------------|
| **ROUGE-like** | Cobertura de contenido | 0.4-0.7 | Mide qué tan bien el resumen representa el contenido original |
| **BLEU Score** | Similitud lexical | 0.3-0.6 | Evalúa la similitud en términos específicos con el original |
| **Coherencia** | Fluidez del resumen | 0.6-1.0 | Indica qué tan bien fluyen las oraciones entre sí |
| **Cobertura** | Frases clave incluidas | 0.7-1.0 | Porcentaje de conceptos importantes capturados |
| **Diversidad** | Variedad lexical | 0.7-0.9 | Mide la riqueza vocabular del resumen |
| **Redundancia** | Repetición de términos | 0.0-0.2 | Menos es mejor - indica repetición excesiva |

## 🎯 **Técnicas de IA Implementadas**

### **TF-ICF (Term Frequency - Inverse Class Frequency)**
Variante especializada de TF-IDF para resumen de documentos individuales. Trata cada oración como una "clase" y calcula la importancia de términos basándose en su distribución entre oraciones.

### **Clustering Semántico con K-means**
Agrupa oraciones similares usando representaciones vectoriales TF-IDF, permitiendo seleccionar oraciones diversas que cubran diferentes temas del documento.

### **Análisis de Frases Clave**
Identifica n-gramas importantes usando TF-IDF a nivel de documento completo, dando mayor peso a oraciones que contienen estos conceptos centrales.

### **Scoring Multi-dimensional**
Combina múltiples señales (posición, longitud, relevancia léxica, frases clave) con pesos aprendidos empíricamente para una selección balanceada.

## ⚡ **Rendimiento y Optimización**

- **Procesamiento CPU**: Optimizado para funcionar sin GPUs
- **Dependencias mínimas**: Solo scikit-learn, numpy y NLTK básico
- **Escalabilidad**: Maneja documentos de 100 a 10,000 palabras
- **Tiempos de procesamiento**: ~1-5 segundos para documentos típicos

## 🔧 **Configuraciones Recomendadas**

### **Para Noticias/Artículos**
```python
pipeline_noticias = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor(min_word_length=2)),
    ('summarizer', SemanticTFICFSummarizer(n_sentences=3))
])
```

### **Para Documentos Técnicos**
```python
pipeline_tecnico = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor(min_word_length=4)),
    ('summarizer', SemanticTFICFSummarizer(n_sentences=4, diversity_weight=0.5))
])
```

### **Para Textos Muy Largos**
```python
pipeline_largo = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor()),
    ('summarizer', SemanticTFICFSummarizer(n_sentences='auto'))
])
```

## 📈 **Resultados Esperados**

Con textos bien estructurados, el sistema típicamente produce:
- **Compresión**: 20-30% del texto original
- **ROUGE Score**: 0.4-0.6
- **BLEU Score**: 0.3-0.5
- **Coherencia**: 0.6-0.8

## 🚨 **Limitaciones y Consideraciones**

- Funciona mejor con textos bien estructurados y párrafos coherentes
- El rendimiento puede variar con textos muy técnicos o especializados
- La detección de idioma asume textos mayoritariamente en un idioma
- Optimizado para español e inglés, otros idiomas requieren ajustes

## 🤝 **Contribuciones**

Las contribuciones son bienvenidas en áreas como:

- Soporte para más idiomas
- Mejoras en la detección de idioma
- Optimizaciones de rendimiento
- Nuevas métricas de evaluación


# 🌐 **API REST - Documentación Completa**

## 🚀 **Instalación y Configuración**

### **Requisitos**
```bash
pip install -r requirements.txt
```

### **Ejecutar el Servidor**
```bash
python main.py
```

La API estará disponible en: `http://localhost:8000`

### **Documentación Interactiva**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📋 **Endpoints Disponibles**

### **1. 🩺 Health Check**
Verifica el estado del servicio y disponibilidad.

**Endpoint**: `GET /api/v1/health`

**Respuesta**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_ready": true,
  "timestamp": "2024-01-15 10:30:45"
}
```

**Ejemplo cURL**:
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

---

### **2. 📝 Resumen de Texto Individual**
Genera un resumen extractivo de un texto individual.

**Endpoint**: `POST /api/v1/summarize`

**Parámetros de Entrada**:
```json
{
  "text": "La inteligencia artificial está transformando radicalmente el panorama tecnológico global. Los avances en machine learning y deep learning han permitido desarrollar sistemas capaces de realizar tareas que antes se consideraban exclusivamente humanas...",
  "n_sentences": 3,
  "language": "auto",
  "include_metrics": true
}
```

**Parámetros**:
- `text` (string, requerido): Texto a resumir (mínimo 100 caracteres)
- `n_sentences` (int, opcional): Número de oraciones en el resumen. `auto` para cálculo automático
- `language` (string, opcional): Idioma del texto. `auto` para detección automática
- `include_metrics` (boolean, opcional): Incluir métricas de evaluación (default: true)

**Respuesta Exitosa**:
```json
{
  "summary": "La inteligencia artificial transforma el panorama tecnológico. Los avances en machine learning permiten sistemas capaces de tareas humanas. Esto impacta diversos sectores como medicina y educación.",
  "original_length": 1250,
  "summary_length": 180,
  "compression_ratio": 0.144,
  "language": "es",
  "selected_sentences": [0, 2, 5],
  "processing_time": 1.23,
  "metrics": {
    "rouge_like_score": 0.65,
    "bleu_score": 0.45,
    "coherence": 0.78,
    "overall_score": 0.72
  },
  "key_phrases": [
    "inteligencia artificial",
    "machine learning", 
    "deep learning",
    "procesamiento lenguaje natural"
  ]
}
```

**Ejemplo cURL**:
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Tu texto largo aquí...",
       "n_sentences": 3,
       "include_metrics": true
     }'
```

---

### **3. 📚 Procesamiento por Lotes**
Genera resúmenes para múltiples textos en una sola solicitud.

**Endpoint**: `POST /api/v1/summarize/batch`

**Parámetros de Entrada**:
```json
{
  "texts": [
    "Primer texto largo para resumir...",
    "Segundo texto largo para resumir...",
    "Tercer texto largo para resumir..."
  ],
  "n_sentences": 3,
  "include_metrics": false
}
```

**Parámetros**:
- `texts` (array, requerido): Lista de textos a resumir (mínimo 1 texto)
- `n_sentences` (int, opcional): Número de oraciones por resumen (default: 3)
- `include_metrics` (boolean, opcional): Incluir métricas para cada texto (default: false)

**Respuesta Exitosa**:
```json
{
  "summaries": [
    {
      "summary": "Resumen del primer texto...",
      "original_length": 800,
      "summary_length": 150,
      "compression_ratio": 0.187,
      "language": "es",
      "selected_sentences": [0, 2, 4],
      "processing_time": 0.0
    },
    {
      "summary": "Resumen del segundo texto...",
      "original_length": 1200,
      "summary_length": 200,
      "compression_ratio": 0.166,
      "language": "es", 
      "selected_sentences": [1, 3, 5],
      "processing_time": 0.0
    }
  ],
  "total_processed": 2,
  "average_compression": 0.176,
  "total_processing_time": 2.45
}
```

**Ejemplo cURL**:
```bash
curl -X POST "http://localhost:8000/api/v1/summarize/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": ["texto1...", "texto2...", "texto3..."],
       "n_sentences": 2,
       "include_metrics": true
     }'
```

---

### **4. 📊 Comparación de Resúmenes**
Evalúa la calidad de un resumen comparándolo con el texto original.

**Endpoint**: `GET /api/v1/metrics/compare`

**Parámetros Query**:
- `original` (string, requerido): Texto original
- `summary` (string, requerido): Texto resumen a evaluar

**Respuesta Exitosa**:
```json
{
  "comparison_metrics": {
    "rouge_like_score": 0.72,
    "bleu_score": 0.51,
    "coherence": 0.85,
    "overall_score": 0.76
  },
  "details": {
    "original_length": 1500,
    "summary_length": 240,
    "compression_ratio": 0.16,
    "key_phrases_coverage": 0.8,
    "coherence_score": 0.85
  },
  "quality_assessment": {
    "quality_level": "BUENA",
    "overall_score": 0.76,
    "issues": ["Podría mejorar la cobertura de contenido"],
    "recommendations": ["Incluir más oraciones clave del texto original"]
  }
}
```

**Ejemplo cURL**:
```bash
curl -X GET "http://localhost:8000/api/v1/metrics/compare?original=TextoOriginal...&summary=ResumenGenerado..."
```

---

### **5. ⚙️ Configuración del Pipeline**
Actualiza parámetros del pipeline en tiempo de ejecución.

**Endpoint**: `PUT /api/v1/config`

**Parámetros Query**:
- `n_sentences` (int, opcional): Número de oraciones para resúmenes (1-20)
- `min_word_length` (int, opcional): Longitud mínima de palabras (2-10)
- `diversity_weight` (float, opcional): Peso para diversidad en scoring (0.0-1.0)

**Respuesta Exitosa**:
```json
{
  "updated_parameters": {
    "n_sentences": 4,
    "min_word_length": 3
  },
  "status": "success"
}
```

**Ejemplo cURL**:
```bash
curl -X PUT "http://localhost:8000/api/v1/config?n_sentences=4&min_word_length=3"
```

---

### **6. 📈 Métricas del Servicio**
Obtiene estadísticas de uso y rendimiento del servicio.

**Endpoint**: `GET /api/v1/metrics/service`

**Respuesta Exitosa**:
```json
{
  "service_uptime": "2h 30m",
  "total_summaries_generated": "N/A",
  "average_processing_time": "N/A", 
  "memory_usage": "N/A"
}
```

**Ejemplo cURL**:
```bash
curl -X GET "http://localhost:8000/api/v1/metrics/service"
```

---

## 🔧 **Ejemplos de Uso en Diferentes Lenguajes**

### **Python**
```python
import requests
import json

# Configurar la URL base
BASE_URL = "http://localhost:8000/api/v1"

# Ejemplo 1: Resumen individual
def resumen_individual(texto):
    payload = {
        "text": texto,
        "n_sentences": 3,
        "include_metrics": True
    }
    
    response = requests.post(f"{BASE_URL}/summarize", json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Ejemplo 2: Procesamiento por lotes
def resumen_lotes(textos):
    payload = {
        "texts": textos,
        "n_sentences": 2,
        "include_metrics": False
    }
    
    response = requests.post(f"{BASE_URL}/summarize/batch", json=payload)
    return response.json()

# Ejemplo de uso
texto_largo = "Tu texto muy largo aquí..."
resultado = resumen_individual(texto_largo)
print(f"Resumen: {resultado['summary']}")
print(f"Compresión: {resultado['compression_ratio']:.1%}")
```

### **JavaScript**
```javascript
// Ejemplo con fetch API
async function generarResumen(texto) {
    const response = await fetch('http://localhost:8000/api/v1/summarize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: texto,
            n_sentences: 3,
            include_metrics: true
        })
    });
    
    const data = await response.json();
    return data;
}

// Uso
const texto = "Texto largo para resumir...";
generarResumen(texto)
    .then(resultado => {
        console.log('Resumen:', resultado.summary);
        console.log('Métricas:', resultado.metrics);
    })
    .catch(error => console.error('Error:', error));
```

### **cURL Avanzado**
```bash
# Resumen con texto desde archivo
curl -X POST "http://localhost:8000/api/v1/summarize" \
     -H "Content-Type: application/json" \
     -d "$(jq -n --arg text "$(cat documento.txt)" '{
       text: $text,
       n_sentences: 4,
       include_metrics: true
     }')"

# Batch processing desde archivo JSON
curl -X POST "http://localhost:8000/api/v1/summarize/batch" \
     -H "Content-Type: application/json" \
     -d @batch_request.json
```

---

## 🚨 **Manejo de Errores**

### **Códigos de Estado HTTP**
- `200 OK`: Solicitud exitosa
- `400 Bad Request`: Parámetros inválidos o texto muy corto
- `500 Internal Server Error`: Error interno del servidor

### **Respuesta de Error**:
```json
{
  "error": "El texto debe tener al menos 100 caracteres",
  "details": {
    "text_length": 45
  },
  "code": 400
}
```

### **Errores Comunes**:
- **Texto muy corto**: El texto debe tener al menos 100 caracteres
- **Texto vacío**: El texto no puede estar vacío
- **Insuficientes oraciones**: El texto debe contener al menos 2 oraciones significativas
- **Parámetros inválidos**: Valores fuera de rangos permitidos

---

## ⚡ **Mejores Prácticas**

### **1. Optimización de Rendimiento**
```python
# Para textos largos (>5000 palabras)
payload = {
    "text": texto_largo,
    "n_sentences": "auto",  # Cálculo automático óptimo
    "include_metrics": False  # Desactivar métricas para mayor velocidad
}
```

### **2. Procesamiento por Lotes Eficiente**
```python
# Agrupar textos similares en tamaño
textos_cortos = [t for t in textos if len(t) < 1000]
textos_largos = [t for t in textos if len(t) >= 1000]

# Procesar por lotes separados
resultados_cortos = resumen_lotes(textos_cortos)
resultados_largos = resumen_lotes(textos_largos)
```

### **3. Manejo de Timeouts**
```python
import requests

# Configurar timeout
try:
    response = requests.post(
        "http://localhost:8000/api/v1/summarize", 
        json=payload, 
        timeout=30  # 30 segundos timeout
    )
except requests.exceptions.Timeout:
    print("La solicitud tardó demasiado tiempo")
```


## 📄 **Licencia**

Este proyecto está bajo la Licencia MIT.