"""
Módulo para el cálculo manual de métricas de evaluación.

Implementa desde cero (sin scikit-learn):
    - Matriz de confusión
    - Precisión, Recall y F1 por clase
    - Accuracy global
    - Macro F1
"""

from collections import defaultdict


def matriz_confusion(y_verdadero, y_predicho, clases):
    """
    Construye una matriz de confusión como diccionario anidado.

    matriz[clase_verdadera][clase_predicha] = cantidad de casos

    Parámetros:
        y_verdadero: lista de etiquetas reales
        y_predicho: lista de etiquetas predichas por el modelo
        clases: lista ordenada de todas las clases posibles

    Retorna:
        dict anidado con las frecuencias
    """
    matriz = {c_real: {c_pred: 0 for c_pred in clases} for c_real in clases}
    for real, pred in zip(y_verdadero, y_predicho):
        matriz[real][pred] += 1
    return matriz


def calcular_metricas_clase(matriz, clase):
    """
    Calcula precisión, recall y F1 para UNA clase específica a partir de la
    matriz de confusión.

    Definiciones:
        TP (true positives):  casos de la clase correctamente predichos
        FP (false positives): casos de OTRAS clases predichos como esta clase
        FN (false negatives): casos de esta clase predichos como OTRAS
    """
    # TP: diagonal de la clase
    tp = matriz[clase][clase]

    # FP: suma de la columna 'clase' excluyendo la diagonal
    # (casos que fueron predichos como 'clase' pero en realidad eran otra)
    fp = sum(matriz[c_real][clase] for c_real in matriz if c_real != clase)

    # FN: suma de la fila 'clase' excluyendo la diagonal
    # (casos que eran realmente 'clase' pero fueron predichos como otra)
    fn = sum(matriz[clase][c_pred] for c_pred in matriz[clase] if c_pred != clase)

    # Precisión: de los predichos como 'clase', cuántos eran realmente
    # Si TP+FP = 0, no predijimos ningún caso de la clase -> precisión indefinida
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: de los que eran realmente 'clase', cuántos detectamos
    # Si TP+FN = 0, no hay casos reales de la clase -> recall indefinido
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1: media armónica. Penaliza más cuando precisión o recall son bajos
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def calcular_accuracy(y_verdadero, y_predicho):
    """
    Accuracy global: proporción de predicciones correctas sobre el total.
    """
    if len(y_verdadero) == 0:
        return 0.0
    aciertos = sum(1 for real, pred in zip(y_verdadero, y_predicho) if real == pred)
    return aciertos / len(y_verdadero)


def reporte_completo(y_verdadero, y_predicho, clases):
    """
    Genera un reporte completo de evaluación con todas las métricas.

    Retorna un diccionario con:
        - matriz_confusion
        - metricas por clase (precision, recall, f1)
        - accuracy global
        - macro_f1 (promedio simple de F1 de todas las clases)
    """
    matriz = matriz_confusion(y_verdadero, y_predicho, clases)

    metricas_por_clase = {}
    for clase in clases:
        metricas_por_clase[clase] = calcular_metricas_clase(matriz, clase)

    accuracy = calcular_accuracy(y_verdadero, y_predicho)

    # Macro F1: promedio simple de los F1 de cada clase.
    # A diferencia del weighted F1, aquí cada clase pesa igual sin importar
    # cuántos ejemplos tenga (importante con clases desbalanceadas).
    macro_f1 = sum(m['f1'] for m in metricas_por_clase.values()) / len(clases)

    return {
        'matriz_confusion': matriz,
        'metricas_por_clase': metricas_por_clase,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
    }


def imprimir_reporte(reporte):
    """
    Imprime el reporte de métricas de forma legible en la terminal.
    """
    print("\n" + "=" * 70)
    print("REPORTE DE EVALUACION")
    print("=" * 70)

    print(f"\nAccuracy global: {reporte['accuracy']:.4f}")
    print(f"Macro F1:        {reporte['macro_f1']:.4f}")

    print(f"\n{'Clase':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Soporte':>10}")
    print("-" * 62)
    for clase, m in reporte['metricas_por_clase'].items():
        soporte = m['tp'] + m['fn']  # cantidad real de ejemplos de esa clase
        print(f"{clase:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {soporte:>10}")


if __name__ == "__main__":
    # Prueba rápida
    y_real = ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B']
    y_pred = ['A', 'B', 'B', 'B', 'C', 'A', 'A', 'C']

    reporte = reporte_completo(y_real, y_pred, ['A', 'B', 'C'])
    imprimir_reporte(reporte)
    print("\nMatriz de confusion:")
    for real in reporte['matriz_confusion']:
        print(f"  Real {real}: {reporte['matriz_confusion'][real]}")