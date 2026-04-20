"""
Script de prueba para ejecutar K-Folds Cross Validation sobre el dataset
real de Bitext. Se ejecuta desde la raíz del proyecto con:
    python -m modelo.probar_kfolds
"""

import pandas as pd
import time
from modelo.procesamiento import Preprocesador
from modelo.kfolds import kfolds_cross_validation


def main():
    print("Cargando dataset...")
    df = pd.read_csv('dataset/bitext_dataset.csv')
    print(f"Total de instancias: {len(df)}")

    # Preprocesamos todos los documentos una sola vez (antes del K-Folds).
    # Esto ahorra mucho tiempo: preprocesar es costoso y los tokens no cambian
    # entre folds, solo cambia qué documentos usamos para entrenar vs probar.
    print("\nPreprocesando documentos...")
    t0 = time.time()
    preprocesador = Preprocesador()
    documentos = [preprocesador.procesar(texto) for texto in df['instruction']]
    etiquetas = df['category'].tolist()
    print(f"Preprocesamiento completo en {time.time() - t0:.1f}s")

    # Ejecutamos K-Folds con K=5 (mínimo que pide el PDF)
    print("\nIniciando K-Folds Cross Validation (K=5)...")
    t0 = time.time()
    resultado = kfolds_cross_validation(documentos, etiquetas, k=5, alpha=1.0)
    print(f"\nTiempo total K-Folds: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()