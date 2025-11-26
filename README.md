# Pipeline de Resumen

Un sistema de resumen extractivo multilingüe implementado como pipeline de scikit-learn que utiliza el algoritmo TF-ICF (Term Frequency - Inverse Class Frequency) para identificar las oraciones más importantes de un texto.

## 🚀 Características

- **Resumen extractivo** basado en importancia semántica
- **Soporte multilingüe** (español e inglés) con detección automática
- **Algoritmo TF-ICF** adaptado para resumen de documentos individuales
- **Pipeline modular** de scikit-learn fácil de extender
- **Preprocesamiento inteligente** con limpieza de texto y stopwords
- **Mínimas dependencias** - solo scikit-learn y numpy

## 📋 Requisitos

```bash
pip install scikit-learn numpy
```

## 🧠 Algoritmo TF-ICF

### Fundamentos Teóricos

El TF-ICF (Term Frequency - Inverse Class Frequency) es una variante del TF-IDF adaptada para tareas de clasificación y resumen:

- **TF (Term Frequency)**: Frecuencia normalizada de términos dentro de una oración
- **ICF (Inverse Class Frequency)**: Medida de qué tan único es un término entre las "clases" (en este caso, oraciones)

### Fórmula Matemática

```
TF(t, s) = (Número de veces que t aparece en s) / (Número total de términos en s)
ICF(t) = log(Total de oraciones / Número de oraciones que contienen t)
Puntaje(s) = Σ [TF(t, s) × ICF(t)] para cada término t en s
```

### Ventajas sobre TF-IDF

- **Mejor para documentos individuales**: TF-ICF trata cada oración como una "clase"
- **Identifica términos discriminativos**: Prioriza palabras que distinguen entre oraciones
- **Óptimo para resumen**: Selecciona oraciones con información única y relevante

## 🛠️ Uso Básico

### Ejemplo Simple

```python
from summarization_pipeline import summarization_pipeline

# Texto a resumir
texto = """
El aprendizaje automático es una rama de la inteligencia artificial. 
Los algoritmos de machine learning permiten a las computadoras aprender patrones en los datos. 
En la actualidad, el deep learning ha revolucionado muchas áreas. 
España es un país con gran desarrollo en tecnología. 
Los investigadores españoles contribuyen significativamente al campo.
"""

# Procesar y obtener resumen
resultados = summarization_pipeline.fit_transform([texto])
resumen = resultados[0]['summary']

print("Resumen:", resumen)
```

### Uso con Múltiples Textos

```python
textos = [
    "Texto en español sobre machine learning...",
    "English text about artificial intelligence...",
    "Otro texto en español sobre deep learning..."
]

resultados = summarization_pipeline.fit_transform(textos)

for i, resultado in enumerate(resultados):
    print(f"Texto {i+1} ({resultado['language']}):")
    print(f"Resumen: {resultado['summary']}")
    print(f"Oraciones seleccionadas: {resultado['selected_sentences']}\n")
```

## 📁 Estructura del Pipeline

### TextPreprocessor

**Responsabilidades:**
- Detección automática de idioma
- División en oraciones
- Limpieza y normalización de texto
- Eliminación de stopwords

**Flujo de procesamiento:**
1. `detect_language()`: Identifica español/inglés por caracteres especiales
2. `split_sentences()`: Divide en oraciones usando regex
3. `preprocess_text()`: Limpia, tokeniza y filtra stopwords

### TFICFSummarizer

**Responsabilidades:**
- Cálculo de scores TF-ICF
- Selección de oraciones relevantes
- Generación del resumen final

**Flujo de cálculo:**
1. `calculate_tf()`: Frecuencia de términos normalizada por oración
2. `calculate_icf()`: Frecuencia inversa entre oraciones
3. `calculate_sentence_scores()`: Combina TF e ICF para puntuar oraciones
4. Selecciona top-N oraciones manteniendo orden original

## ⚙️ Personalización

### Modificar Número de Oraciones

```python
pipeline_personalizado = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('summarizer', TFICFSummarizer(n_sentences=3))  # 3 oraciones en el resumen
])
```

### Agregar Stopwords Personalizadas

```python
class TextPreprocessorPersonalizado(TextPreprocessor):
    def __init__(self):
        super().__init__()
        # Agregar stopwords personalizadas
        self.stopwords_es.update({'python', 'código', 'programación'})
        self.stopwords_en.update({'python', 'code', 'programming'})
```

### Pipeline para Dominio Específico

```python
class DomainSpecificSummarizer(TFICFSummarizer):
    def __init__(self, n_sentences=2, domain_terms=None):
        super().__init__(n_sentences)
        self.domain_terms = domain_terms or {}
    
    def calculate_sentence_scores(self, processed_data):
        scores = super().calculate_sentence_scores(processed_data)
        # Bonus para términos del dominio
        for i, (idx, score, length) in enumerate(scores):
            domain_bonus = self._calculate_domain_bonus(processed_data['processed_sentences'][idx])
            scores[i] = (idx, score * (1 + domain_bonus), length)
        return scores
    
    def _calculate_domain_bonus(self, sentence):
        # Implementar lógica de bonus para términos del dominio
        pass
```

## 📊 Ejemplos Completos

### Ejemplo 1: Texto Científico

```python
texto_cientifico = """
La inteligencia artificial está transformando la investigación científica. 
Los modelos de deep learning pueden predecir estructuras proteicas con alta precisión. 
Estos avances aceleran el desarrollo de nuevos medicamentos. 
Sin embargo, existen desafíos éticos en el uso de IA en medicina. 
La interpretabilidad de los modelos sigue siendo un problema importante.
"""

resultado = summarization_pipeline.fit_transform([texto_cientifico])[0]
print(f"Idioma: {resultado['language']}")
print(f"Resumen: {resultado['summary']}")
print(f"Oraciones seleccionadas: {resultado['selected_sentences']}")
```

### Ejemplo 2: Texto Periodístico

```python
texto_noticia = """
El cambio climático afecta gravemente a los ecosistemas marinos. 
Las temperaturas oceánicas han aumentado significativamente en la última década. 
Esto provoca la decoloración de los arrecifes de coral en todo el mundo. 
Los científicos advierten sobre consecuencias irreversibles si no se toman medidas. 
Varios países han firmado acuerdos para reducir las emisiones de carbono.
"""

resultado = summarization_pipeline.fit_transform([texto_noticia])[0]
```

## 🧪 Testing y Validación

### Ejecutar Ejemplos de Prueba

```bash
python summarization_pipeline.py
```

### Output Esperado

```
--- Texto 1 (ES) ---
Original:
    El aprendizaje automático es una rama de la inteligencia artificial. 
    Los algoritmos de machine...

Resumen:
Los algoritmos de machine learning permiten a las computadoras aprender patrones en los datos. En la actualidad, el deep learning ha revolucionado muchas áreas.
Oraciones seleccionadas: [1, 2]
--------------------------------------------------
```

## 🔧 Extensión del Sistema

### Agregar Nuevos Idiomas

```python
class MultilingualTextPreprocessor(TextPreprocessor):
    def __init__(self):
        super().__init__()
        self.stopwords_fr = {'le', 'la', 'de', 'et', 'à'}  # Francés
        self.stopwords_pt = {'o', 'a', 'de', 'e', 'em'}   # Portugués
    
    def detect_language(self, text):
        # Implementar detección más sofisticada
        if re.search(r'[áéíóúñ]', text):
            return 'es'
        elif re.search(r'[àâêîôû]', text):
            return 'fr'
        else:
            return 'en'
```

### Integración con APIs Externas

```python
class APISummarizer(TFICFSummarizer):
    def __init__(self, n_sentences=2, api_key=None):
        super().__init__(n_sentences)
        self.api_key = api_key
    
    def transform(self, X):
        resultados = super().transform(X)
        # Enriquecer resultados con API externa
        for resultado in resultados:
            resultado['entities'] = self._extract_entities(resultado['summary'])
        return resultados
```

## 📈 Métricas y Evaluación

### Evaluación de Calidad

```python
def evaluate_summary_quality(original, summary, reference_summary=None):
    """Evalúa la calidad del resumen usando métricas simples"""
    
    # Métricas básicas
    compression_ratio = len(summary) / len(original)
    sentence_reduction = 1 - (summary.count('.') / original.count('.'))
    
    metrics = {
        'compression_ratio': compression_ratio,
        'sentence_reduction': sentence_reduction,
        'summary_length': len(summary),
        'original_length': len(original)
    }
    
    return metrics
```

## 🐛 Solución de Problemas

### Problemas Comunes

1. **Oraciones vacías en el resumen**
   - Causa: Preprocesamiento muy agresivo
   - Solución: Ajustar umbral de stopwords o longitud mínima

2. **Detección incorrecta de idioma**
   - Causa: Textos mixtos o sin caracteres especiales
   - Solución: Implementar detección más robusta

3. **Resumen muy corto/largo**
   - Causa: Parámetro n_sentences inadecuado
   - Solución: Ajustar dinámicamente según longitud del texto

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Áreas de mejora:

- [ ] Soporte para más idiomas
- [ ] Detección de idioma más robusta
- [ ] Integración con modelos transformer
- [ ] Evaluación automática de calidad
- [ ] Interfaz web o API REST

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

