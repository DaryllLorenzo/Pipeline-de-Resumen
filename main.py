from sklearn.pipeline import Pipeline
from text_preprocessor import EnhancedTextPreprocessor
from semantic_summarizer import SemanticTFICFSummarizer
from metrics_evaluator import AdvancedSummaryEvaluator

# Pipeline principal
semantic_pipeline = Pipeline([
    ('preprocessor', EnhancedTextPreprocessor()),
    ('summarizer', SemanticTFICFSummarizer(n_sentences='auto', clustering_method='kmeans'))
])

# Pipeline por defecto
summarization_pipeline = semantic_pipeline

def main():
    """Función principal con ejemplos de uso."""
    
    # Textos de ejemplo (el mismo que tenías)
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
    Many experts argue that while AI will eliminate some routine jobs, it will also create new employment opportunities in emerging fields. 
    La clave para navegar esta transición será la educación continua y el desarrollo de habilidades digitales en la fuerza laboral. 
    Las empresas y gobiernos deben colaborar para asegurar que los beneficios de la inteligencia artificial sean distribuidos equitativamente en la sociedad.
    """
    
    english_long_text = """
    Artificial intelligence is fundamentally reshaping the global technological landscape. 
    Breakthroughs in machine learning and deep learning have enabled the development of systems capable of performing tasks once considered exclusively human. 
    In the medical field, AI algorithms can analyze medical images with accuracy that rivals expert radiologists. 
    This has led to earlier and more precise diagnoses of diseases like cancer, significantly improving survival rates. 
    However, the implementation of these technologies faces significant challenges regarding data privacy and ethics. 
    Protecting sensitive medical information is a paramount concern that requires robust regulatory frameworks. 
    In the financial sector, AI systems are revolutionizing fraud detection and risk management. 
    Algorithms can analyze millions of transactions in real-time, identifying suspicious patterns that would be imperceptible to human analysts. 
    This capability has reduced fraud losses in financial institutions by over 30% according to recent studies. 
    Education is another area experiencing profound transformation thanks to artificial intelligence. 
    Adaptive learning systems can customize educational content according to the individual needs of each student. 
    This is proving particularly effective for closing educational gaps and improving academic performance in diverse populations. 
    Despite these promising advances, there are legitimate concerns about AI's impact on employment. 
    Many experts argue that while AI will eliminate some routine jobs, it will also create new employment opportunities in emerging fields. 
    The key to navigating this transition will be continuous education and development of digital skills in the workforce. 
    Businesses and governments must collaborate to ensure that the benefits of artificial intelligence are distributed equitably across society.
    """
    
    print("=== SISTEMA DE RESUMEN AVANZADO ===")
    print("Pipeline modular con evaluación de métricas")
    print("=" * 60)
    
    evaluator = AdvancedSummaryEvaluator()
    evaluations = []
    
    # Procesar textos
    test_texts = [
        ("Español Largo", spanish_long_text),
        ("Inglés Largo", english_long_text)
    ]
    
    for text_name, text in test_texts:
        print(f"\n📖 Procesando: {text_name}")
        print("-" * 40)
        
        try:
            # Generar resumen
            results = semantic_pipeline.fit_transform([text])
            if results:
                result = results[0]
                
                print(f"✅ Resumen generado ({len(result['selected_sentences'])} oraciones):")
                print(f"\"{result['summary']}\"")
                print(f"📊 Compresión: {result['compression_ratio']:.1%}")
                
                # Preparar datos para métricas
                # El resultado ya contiene key_phrases gracias a la modificación en semantic_summarizer
                processed_data_for_metrics = {
                    'key_phrases': result.get('key_phrases', []),
                    'sentences': result.get('sentences', []),
                    'original': result['original']
                }
                
                # Evaluar con métricas
                evaluation = evaluator.comprehensive_evaluation(
                    text, 
                    result['summary'], 
                    f"Semantic TF-ICF - {text_name}",
                    processed_data_for_metrics,  # Pasar datos procesados
                    result['selected_sentences']
                )
                evaluations.append(evaluation)
                
                print(f"🎯 Score general: {evaluation['metrics']['overall_score']:.4f}")
                print(f"📈 ROUGE-like: {evaluation['metrics']['rouge_like_score']:.4f}")
                print(f"🔤 BLEU: {evaluation['metrics']['bleu_score']:.4f}")
                
        except Exception as e:
            print(f"❌ Error procesando {text_name}: {e}")
            continue
    
    # Mostrar análisis comparativo
    if evaluations:
        print("\n" + "="*80)
        print("RESUMEN EJECUTIVO - COMPARACIÓN FINAL")
        print("="*80)
        evaluator.print_detailed_analysis(evaluations)
    else:
        print("\n❌ No se pudieron generar evaluaciones.")

if __name__ == "__main__":
    main()