"""
Archivo de pruebas y ejemplos del pipeline de resumen.
Para usar la API, ejecutar main.py y usar los endpoints.
"""

from sklearn.pipeline import Pipeline
from text_preprocessor import EnhancedTextPreprocessor
from semantic_summarizer import SemanticTFICFSummarizer
from metrics_evaluator import AdvancedSummaryEvaluator

# Pipeline principal (mismo que antes)
semantic_pipeline = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor()),
    ('summarizer', SemanticTFICFSummarizer(n_sentences='auto', clustering_method='kmeans'))
])

def test_pipeline():
    """Función de prueba del pipeline local"""
    
    spanish_long_text = """
    La inteligencia artificial está transformando radicalmente el panorama tecnológico global. 
    Los avances en machine learning y deep learning han permitido desarrollar sistemas capaces de realizar tareas que antes se consideraban exclusivamente humanas. 
    En el campo de la medicina, los algoritmos de IA pueden analizar imágenes médicas con una precisión que rivaliza con la de radiólogos expertos. 
    Esto ha llevado a diagnósticos más tempranos y precisos de enfermedades como el cáncer, mejorando significativamente las tasas de supervivencia. 
    Sin embargo, la implementación de estas tecnologías enfrenta desafíos significativos en cuanto a privacidad de datos y ética. 
    La protección de la información médica sensible es una preocupación primordial que requiere marcos regulatorios robustos. 
    En el sector financiero, los sistemas de IA están revolucionando la detección de fraudes y la gestión de riesgos. 
    Los algoritmos pueden analizar millones de transacciones en tiempo real, identificando patrones sospechosos que serían imperceptibles para los analistas humanos. 
    Esta capacidad ha reducido las pérdidas por fraude en instituciones financieras en más de un 30% según estudios recientes. 
    La educación es otra área que está experimentando una transformación profunda gracias a la inteligencia artificial. 
    Los sistemas de aprendizaje adaptativo pueden personalizar el contenido educativo según las necesidades individuales de cada estudiante. 
    Esto está demostrando ser particularmente efectivo para cerrar brechas educativas y mejorar el rendimiento académico en poblaciones diversas. 
    A pesar de estos avances prometedores, existen preocupaciones legítimas sobre el impacto de la IA en el empleo. 
    Muchos expertos argumentan que, aunque la IA eliminará algunos trabajos rutinarios, también creará nuevas oportunidades laborales en campos emergentes.
    La clave para navegar esta transición será la educación continua y el desarrollo de habilidades digitales en la fuerza laboral. 
    Las empresas y gobiernos deben colaborar para asegurar que los beneficios de la inteligencia artificial sean distribuidos equitativamente en la sociedad.
    """
    
    print("=== PRUEBA LOCAL DEL PIPELINE ===")
    print("(Para usar la API, ejecutar: python main.py)")
    print("=" * 50)
    
    evaluator = AdvancedSummaryEvaluator()
    
    try:
        # Generar resumen
        results = semantic_pipeline.fit_transform([spanish_long_text])
        
        if results:
            result = results[0]
            
            print(f"✅ Resumen generado ({len(result['selected_sentences'])} oraciones):")
            print(f"\"{result['summary']}\"")
            print(f"📊 Compresión: {result['compression_ratio']:.1%}")
            print(f"🔤 Idioma: {result['language']}")
            
            # Evaluar con métricas
            processed_data_for_metrics = {
                'key_phrases': result.get('key_phrases', []),
                'sentences': result.get('sentences', []),
                'original': result['original']
            }
            
            evaluation = evaluator.comprehensive_evaluation(
                spanish_long_text, 
                result['summary'], 
                "Prueba Local",
                processed_data_for_metrics,
                result['selected_sentences']
            )
            
            print(f"🎯 Score general: {evaluation['metrics']['overall_score']:.4f}")
            print(f"📈 ROUGE-like: {evaluation['metrics']['rouge_like_score']:.4f}")
            print(f"🔤 BLEU: {evaluation['metrics']['bleu_score']:.4f}")
            print(f"🔄 Coherencia: {evaluation['metrics']['coherence']:.4f}")
            
    except Exception as e:
        print(f"❌ Error en prueba local: {e}")

def test_api_example():
    """Ejemplo de cómo usar la API una vez ejecutada"""
    print("\n" + "="*50)
    print("EJEMPLO DE USO DE LA API:")
    print("1. Ejecutar: python main.py")
    print("2. Abrir: http://localhost:8000/docs")
    print("3. Usar endpoints:")
    print("   - POST /api/v1/summarize")
    print("   - POST /api/v1/summarize/batch")
    print("   - GET  /api/v1/metrics/compare")
    print("="*50)

if __name__ == "__main__":
    test_pipeline()
    test_api_example()