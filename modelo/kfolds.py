"""
Implementación manual de K-Folds Cross Validation.

No se utiliza sklearn.model_selection.KFold. Todo el particionado y
promediado se hace manualmente.
"""

import random
from collections import defaultdict
from modelo.naive_bayes import NaiveBayesMultinomial
from modelo.metricas import reporte_completo, imprimir_reporte


def dividir_en_folds(indices, k, seed=42):
    """
    Divide una lista de índices en K partes aproximadamente iguales.

    Mezclamos aleatoriamente primero para que cada fold tenga una mezcla
    balanceada de clases (no queden folds con una sola clase).

    Usamos seed fijo para que los resultados sean reproducibles entre corridas.
    """
    indices_mezclados = indices.copy()
    random.Random(seed).shuffle(indices_mezclados)

    tamano_fold = len(indices_mezclados) // k
    folds = []
    for i in range(k):
        inicio = i * tamano_fold
        # Último fold se lleva los sobrantes de la división entera
        fin = (i + 1) * tamano_fold if i < k - 1 else len(indices_mezclados)
        folds.append(indices_mezclados[inicio:fin])
    return folds


def kfolds_cross_validation(documentos, etiquetas, k=5, alpha=1.0, verbose=True):
    """
    Ejecuta K-Folds Cross Validation completo.

    Para cada uno de los K experimentos:
        1. Separa un fold como conjunto de prueba
        2. Usa los otros K-1 folds como entrenamiento
        3. Entrena un modelo nuevo desde cero
        4. Evalúa sobre el fold de prueba
        5. Guarda las métricas

    Al final promedia y calcula varianza entre folds.

    Parámetros:
        documentos: lista de listas de tokens ya preprocesadas
        etiquetas: lista de strings con las clases
        k: número de folds (mínimo 5 según el PDF del proyecto)
        alpha: parámetro de Laplace Smoothing
        verbose: si imprime progreso en terminal

    Retorna:
        dict con resultados agregados de los K experimentos
    """
    if len(documentos) != len(etiquetas):
        raise ValueError("documentos y etiquetas deben tener la misma longitud")

    n = len(documentos)
    clases = sorted(set(etiquetas))
    indices = list(range(n))
    folds = dividir_en_folds(indices, k)

    resultados_por_fold = []

    for i in range(k):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"FOLD {i + 1}/{k}")
            print(f"{'=' * 60}")

        # El fold i es el de prueba, los demás son de entrenamiento
        indices_prueba = folds[i]
        indices_entrenamiento = [idx for j, fold in enumerate(folds) if j != i for idx in fold]

        # Separamos documentos y etiquetas para este experimento
        docs_train = [documentos[idx] for idx in indices_entrenamiento]
        etiq_train = [etiquetas[idx] for idx in indices_entrenamiento]
        docs_test = [documentos[idx] for idx in indices_prueba]
        etiq_test = [etiquetas[idx] for idx in indices_prueba]

        if verbose:
            print(f"Tamano entrenamiento: {len(docs_train)}")
            print(f"Tamano prueba:        {len(docs_test)}")

        # Entrenamos modelo nuevo desde cero
        modelo = NaiveBayesMultinomial(alpha=alpha)
        modelo.entrenar(docs_train, etiq_train)

        # Predecimos sobre el conjunto de prueba
        predicciones = modelo.predecir_batch(docs_test)

        # Calculamos métricas
        reporte = reporte_completo(etiq_test, predicciones, clases)
        resultados_por_fold.append(reporte)

        if verbose:
            print(f"Accuracy: {reporte['accuracy']:.4f} | Macro F1: {reporte['macro_f1']:.4f}")

    # =====================================================
    # Agregación de resultados entre folds
    # =====================================================
    accuracies = [r['accuracy'] for r in resultados_por_fold]
    macro_f1s = [r['macro_f1'] for r in resultados_por_fold]

    accuracy_promedio = sum(accuracies) / k
    macro_f1_promedio = sum(macro_f1s) / k

    # Varianza: qué tanto varían los resultados entre folds.
    # Varianza baja -> modelo estable. Varianza alta -> el modelo es sensible
    # a qué datos usó para entrenar (posible sobreajuste o datos ruidosos).
    varianza_accuracy = sum((a - accuracy_promedio) ** 2 for a in accuracies) / k
    varianza_macro_f1 = sum((f - macro_f1_promedio) ** 2 for f in macro_f1s) / k

    # Desviación estándar (raíz de la varianza, misma unidad que la métrica)
    desv_accuracy = varianza_accuracy ** 0.5
    desv_macro_f1 = varianza_macro_f1 ** 0.5

    # Promedio de precisión, recall y F1 por clase a través de los folds
    metricas_clase_promedio = {c: {'precision': 0, 'recall': 0, 'f1': 0} for c in clases}
    for reporte in resultados_por_fold:
        for clase in clases:
            metricas_clase_promedio[clase]['precision'] += reporte['metricas_por_clase'][clase]['precision']
            metricas_clase_promedio[clase]['recall'] += reporte['metricas_por_clase'][clase]['recall']
            metricas_clase_promedio[clase]['f1'] += reporte['metricas_por_clase'][clase]['f1']
    for clase in clases:
        for metrica in metricas_clase_promedio[clase]:
            metricas_clase_promedio[clase][metrica] /= k

    # Matriz de confusión acumulada de los K folds (suma casilla por casilla)
    matriz_acumulada = {c_real: {c_pred: 0 for c_pred in clases} for c_real in clases}
    for reporte in resultados_por_fold:
        for c_real in clases:
            for c_pred in clases:
                matriz_acumulada[c_real][c_pred] += reporte['matriz_confusion'][c_real][c_pred]

    resultado_final = {
        'k': k,
        'clases': clases,
        'accuracies_por_fold': accuracies,
        'macro_f1s_por_fold': macro_f1s,
        'accuracy_promedio': accuracy_promedio,
        'macro_f1_promedio': macro_f1_promedio,
        'desv_accuracy': desv_accuracy,
        'desv_macro_f1': desv_macro_f1,
        'varianza_accuracy': varianza_accuracy,
        'varianza_macro_f1': varianza_macro_f1,
        'metricas_clase_promedio': metricas_clase_promedio,
        'matriz_acumulada': matriz_acumulada,
    }

    if verbose:
        imprimir_resumen_kfolds(resultado_final)

    return resultado_final


def imprimir_resumen_kfolds(resultado):
    """
    Imprime un resumen con los promedios y la varianza entre folds.
    """
    k = resultado['k']
    print("\n" + "=" * 70)
    print(f"RESUMEN K-FOLDS CROSS VALIDATION (K={k})")
    print("=" * 70)

    print("\nResultados por fold:")
    for i in range(k):
        print(f"  Fold {i+1}: Accuracy = {resultado['accuracies_por_fold'][i]:.4f} | "
              f"Macro F1 = {resultado['macro_f1s_por_fold'][i]:.4f}")

    print(f"\nAccuracy promedio:  {resultado['accuracy_promedio']:.4f} "
          f"(+/- {resultado['desv_accuracy']:.4f})")
    print(f"Macro F1 promedio:  {resultado['macro_f1_promedio']:.4f} "
          f"(+/- {resultado['desv_macro_f1']:.4f})")
    print(f"Varianza accuracy:  {resultado['varianza_accuracy']:.6f}")
    print(f"Varianza macro F1:  {resultado['varianza_macro_f1']:.6f}")

    print(f"\n{'Clase':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    for clase, m in resultado['metricas_clase_promedio'].items():
        print(f"{clase:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")