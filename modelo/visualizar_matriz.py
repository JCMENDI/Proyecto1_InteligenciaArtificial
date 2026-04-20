"""
Genera un heatmap de la matriz de confusión usando los resultados
acumulados del K-Folds. La imagen se guarda en documentacion/capturas/
para usarla en el informe PDF.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from modelo.procesamiento import Preprocesador
from modelo.kfolds import kfolds_cross_validation


def matriz_a_dataframe(matriz_dict, clases):
    """
    Convierte el dict anidado de la matriz de confusion en un DataFrame
    para poder graficarlo con seaborn.
    """
    datos = np.zeros((len(clases), len(clases)), dtype=int)
    for i, c_real in enumerate(clases):
        for j, c_pred in enumerate(clases):
            datos[i][j] = matriz_dict[c_real][c_pred]
    return pd.DataFrame(datos, index=clases, columns=clases)


def graficar_matriz(matriz_df, titulo, ruta_salida, normalizar=False):
    """
    Dibuja la matriz de confusion como heatmap.

    Parametros:
        normalizar: si True, muestra proporciones por fila (recall visual).
                    Util para comparar clases con distinto numero de ejemplos.
    """
    if normalizar:
        # Normalizamos por fila: cada fila suma 1
        # Asi la diagonal muestra directamente el recall de cada clase
        matriz_plot = matriz_df.div(matriz_df.sum(axis=1), axis=0)
        fmt = '.2f'
        cmap = 'Blues'
    else:
        matriz_plot = matriz_df
        fmt = 'd'
        cmap = 'Blues'

    plt.figure(figsize=(12, 10))
    sns.heatmap(matriz_plot,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                cbar_kws={'label': 'Proporcion' if normalizar else 'Cantidad'},
                linewidths=0.5,
                linecolor='gray')
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.ylabel('Clase Real', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardada: {ruta_salida}")


def main():
    print("Cargando y preprocesando dataset...")
    df = pd.read_csv('dataset/bitext_dataset.csv')
    preprocesador = Preprocesador()
    documentos = [preprocesador.procesar(t) for t in df['instruction']]
    etiquetas = df['category'].tolist()

    print("\nEjecutando K-Folds para obtener matriz acumulada...")
    # verbose=False para no duplicar la salida
    resultado = kfolds_cross_validation(documentos, etiquetas, k=5, alpha=1.0, verbose=False)

    matriz_df = matriz_a_dataframe(resultado['matriz_acumulada'], resultado['clases'])

    print("\nGenerando visualizaciones...")
    graficar_matriz(
        matriz_df,
        'Matriz de Confusion - K-Folds Acumulada (frecuencias)',
        'documentacion/capturas/matriz_confusion_frecuencias.png',
        normalizar=False
    )
    graficar_matriz(
        matriz_df,
        'Matriz de Confusion - K-Folds Acumulada (normalizada por fila)',
        'documentacion/capturas/matriz_confusion_normalizada.png',
        normalizar=True
    )

    # Guardamos tambien la matriz como CSV por si queremos incluirla en el informe
    matriz_df.to_csv('documentacion/capturas/matriz_confusion.csv')
    print("  Guardada: documentacion/capturas/matriz_confusion.csv")

    # Imprimimos las confusiones mas notables para el analisis
    print("\n" + "=" * 60)
    print("CONFUSIONES MAS NOTABLES (para el analisis critico)")
    print("=" * 60)
    confusiones = []
    for c_real in resultado['clases']:
        for c_pred in resultado['clases']:
            if c_real != c_pred:
                n = resultado['matriz_acumulada'][c_real][c_pred]
                if n > 0:
                    confusiones.append((c_real, c_pred, n))
    confusiones.sort(key=lambda x: -x[2])
    for c_real, c_pred, n in confusiones[:10]:
        print(f"  {c_real:<15} -> {c_pred:<15} : {n} casos")


if __name__ == "__main__":
    main()