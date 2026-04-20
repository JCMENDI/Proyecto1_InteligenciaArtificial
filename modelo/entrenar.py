"""
Script de entrenamiento final del modelo.

A diferencia del K-Folds (que entrena con el 80% para evaluar),
aquí entrenamos con el 100% del dataset para obtener el modelo definitivo
que se usará en producción (en la aplicación web).

El modelo entrenado se guarda con pickle en modelo/modelo_entrenado.pkl.
"""

import pandas as pd
import pickle
import time
import os

from modelo.procesamiento import Preprocesador
from modelo.naive_bayes import NaiveBayesMultinomial


def entrenar_y_guardar(ruta_dataset='dataset/bitext_dataset.csv',
                       ruta_salida='modelo/modelo_entrenado.pkl',
                       alpha=1.0):
    """
    Entrena el clasificador Naïve Bayes con el dataset completo
    y lo persiste en disco junto con su preprocesador.

    Guardamos TAMBIÉN el preprocesador porque la aplicación web necesita
    aplicar exactamente las mismas transformaciones al texto del usuario
    antes de pasarlo al clasificador. Si no, los tokens no coincidirían
    con el vocabulario del modelo.
    """
    print("=" * 60)
    print("ENTRENAMIENTO FINAL DEL MODELO")
    print("=" * 60)

    # 1. Cargar dataset
    print(f"\nCargando dataset desde {ruta_dataset}...")
    df = pd.read_csv(ruta_dataset)
    print(f"  Total de instancias: {len(df)}")
    print(f"  Categorias: {df['category'].nunique()}")

    # 2. Preprocesar todos los documentos
    print("\nPreprocesando documentos...")
    t0 = time.time()
    preprocesador = Preprocesador()
    documentos = [preprocesador.procesar(texto) for texto in df['instruction']]
    etiquetas = df['category'].tolist()
    print(f"  Completado en {time.time() - t0:.1f}s")

    # 3. Entrenar el modelo con todos los datos
    print("\nEntrenando Naive Bayes...")
    t0 = time.time()
    modelo = NaiveBayesMultinomial(alpha=alpha)
    modelo.entrenar(documentos, etiquetas)
    print(f"  Completado en {time.time() - t0:.1f}s")
    print(f"  Tamano del vocabulario: {len(modelo.vocabulario)}")
    print(f"  Clases: {modelo.clases}")

    # 4. Guardar modelo Y preprocesador juntos en el mismo archivo
    # De esta forma, al cargar el pickle obtenemos todo lo necesario
    # para clasificar texto nuevo sin tener que reentrenar.
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    paquete = {
        'modelo': modelo,
        'preprocesador': preprocesador,
        'metadata': {
            'n_instancias': len(df),
            'n_clases': len(modelo.clases),
            'tamano_vocabulario': len(modelo.vocabulario),
            'alpha': alpha,
        }
    }
    with open(ruta_salida, 'wb') as f:
        pickle.dump(paquete, f)

    # Tamaño del archivo en MB para el informe
    tamano_mb = os.path.getsize(ruta_salida) / (1024 * 1024)
    print(f"\nModelo guardado en: {ruta_salida}")
    print(f"  Tamano: {tamano_mb:.2f} MB")

    return modelo, preprocesador


def cargar_modelo(ruta='modelo/modelo_entrenado.pkl'):
    """
    Carga el modelo y el preprocesador desde el archivo pickle.
    Esta funcion la va a usar la aplicacion web.
    """
    with open(ruta, 'rb') as f:
        paquete = pickle.load(f)
    return paquete['modelo'], paquete['preprocesador'], paquete['metadata']


if __name__ == "__main__":
    modelo, preprocesador = entrenar_y_guardar()

    # Prueba de carga para verificar que el pickle funciona
    print("\n" + "=" * 60)
    print("VERIFICACION DE CARGA")
    print("=" * 60)
    modelo_cargado, preproc_cargado, meta = cargar_modelo()
    print(f"  Modelo cargado correctamente")
    print(f"  Metadata: {meta}")

    # Prueba de prediccion con ejemplos reales
    pruebas = [
        "I want to cancel my order",
        "How do I get a refund for my purchase?",
        "Where is my package? It has not arrived yet",
        "I need to update my billing information",
    ]

    print("\nPredicciones de prueba:")
    for texto in pruebas:
        tokens = preproc_cargado.procesar(texto)
        clase, probs = modelo_cargado.predecir_con_probabilidades(tokens)
        confianza = probs[clase] * 100
        print(f'\n  "{texto}"')
        print(f"    -> {clase} ({confianza:.1f}% de confianza)")