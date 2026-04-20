"""
Implementación desde cero del algoritmo Naïve Bayes Multinomial
para clasificación de texto.

No se utilizan librerías de Machine Learning (scikit-learn, etc.).
Todo el cálculo se realiza manualmente usando estructuras nativas de Python.

Conceptos implementados:
    - Bag of Words (vocabulario y conteo de frecuencias por clase)
    - Probabilidades a priori por clase
    - Laplace Smoothing (add-one) para manejar palabras no vistas
    - Suma de logaritmos durante la inferencia para evitar underflow numérico
"""

import math
from collections import defaultdict, Counter


class NaiveBayesMultinomial:
    """
    Clasificador Naïve Bayes Multinomial.

    Funciona con documentos representados como listas de tokens (palabras
    ya preprocesadas). Se entrena con el método entrenar() y predice con
    el método predecir().
    """

    def __init__(self, alpha=1.0):
        """
        Parámetros:
            alpha: parámetro de Laplace Smoothing. Con alpha=1 es el clásico
                   "add-one smoothing". Valores menores hacen menos suavizado.
        """
        self.alpha = alpha

        # Estructuras que se llenan durante el entrenamiento
        self.vocabulario = set()           # conjunto de palabras únicas vistas
        self.clases = []                   # lista de clases posibles
        self.log_prior = {}                # log P(C) para cada clase C
        self.log_verosimilitud = {}        # log P(w|C) para cada palabra w y clase C
        self.conteo_palabras_clase = {}    # total de palabras (con repetición) por clase
        self.entrenado = False

    def entrenar(self, documentos, etiquetas):
        """
        Entrena el modelo a partir de documentos ya preprocesados.

        Parámetros:
            documentos: lista de listas de tokens. Ejemplo:
                        [['need', 'cancel', 'order'], ['refund', 'money'], ...]
            etiquetas: lista de strings con la clase de cada documento.
                       Debe tener la misma longitud que documentos.
        """
        if len(documentos) != len(etiquetas):
            raise ValueError("documentos y etiquetas deben tener la misma longitud")

        n_documentos = len(documentos)
        self.clases = sorted(set(etiquetas))

        # =============================================================
        # PASO 1: Construir el vocabulario global (Bag of Words)
        # =============================================================
        # Recorremos todos los documentos y acumulamos cada palabra única.
        # Este vocabulario se usa en Laplace Smoothing como |V|.
        for tokens in documentos:
            self.vocabulario.update(tokens)

        tamano_vocabulario = len(self.vocabulario)

        # =============================================================
        # PASO 2: Contar documentos y palabras por clase
        # =============================================================
        # documentos_por_clase[C] = cuántos documentos pertenecen a C
        # frecuencia_por_clase[C][w] = cuántas veces aparece w en documentos de C
        # conteo_palabras_clase[C] = suma total de palabras en documentos de C
        documentos_por_clase = defaultdict(int)
        frecuencia_por_clase = defaultdict(Counter)
        self.conteo_palabras_clase = defaultdict(int)

        for tokens, clase in zip(documentos, etiquetas):
            documentos_por_clase[clase] += 1
            for token in tokens:
                frecuencia_por_clase[clase][token] += 1
                self.conteo_palabras_clase[clase] += 1

        # =============================================================
        # PASO 3: Calcular probabilidades a priori log P(C)
        # =============================================================
        # P(C) = (cantidad de documentos de clase C) / (total de documentos)
        # Usamos log() directamente para evitar calcular P y después log P.
        for clase in self.clases:
            prior = documentos_por_clase[clase] / n_documentos
            self.log_prior[clase] = math.log(prior)

        # =============================================================
        # PASO 4: Calcular verosimilitudes log P(w|C) con Laplace Smoothing
        # =============================================================
        # Formula:
        #     P(w|C) = (count(w, C) + alpha) / (total_palabras_C + alpha * |V|)
        #
        # Precalculamos el log de cada P(w|C) para cada palabra del vocabulario
        # en cada clase. Esto hace la inferencia mucho más rápida después.
        self.log_verosimilitud = {clase: {} for clase in self.clases}

        for clase in self.clases:
            # Denominador: total de palabras en la clase + suavizado global
            denominador = self.conteo_palabras_clase[clase] + self.alpha * tamano_vocabulario

            for palabra in self.vocabulario:
                # Numerador: cuántas veces aparece la palabra en la clase + alpha
                numerador = frecuencia_por_clase[clase][palabra] + self.alpha
                self.log_verosimilitud[clase][palabra] = math.log(numerador / denominador)

        self.entrenado = True

    def _log_probabilidad_clase(self, tokens, clase):
        """
        Calcula log P(C|X) ∝ log P(C) + Σ log P(w|C) para cada palabra del documento.

        Notas importantes:
            - Palabras que no están en el vocabulario se ignoran
              (es la convención estándar en Naïve Bayes Multinomial).
            - Palabras que sí están en el vocabulario ya tienen su log P(w|C)
              precalculado con Laplace Smoothing, entonces nunca dan -inf.
        """
        log_prob = self.log_prior[clase]

        for token in tokens:
            if token in self.vocabulario:
                log_prob += self.log_verosimilitud[clase][token]
            # else: palabra nunca vista en entrenamiento -> se ignora

        return log_prob

    def predecir(self, tokens):
        """
        Predice la clase de un documento (lista de tokens preprocesados).
        Retorna el nombre de la clase con mayor log-probabilidad.
        """
        if not self.entrenado:
            raise RuntimeError("El modelo no ha sido entrenado. Llamar entrenar() primero.")

        mejor_clase = None
        mejor_log_prob = -math.inf

        for clase in self.clases:
            log_prob = self._log_probabilidad_clase(tokens, clase)
            if log_prob > mejor_log_prob:
                mejor_log_prob = log_prob
                mejor_clase = clase

        return mejor_clase

    def predecir_con_probabilidades(self, tokens):
        """
        Igual que predecir() pero además retorna las probabilidades normalizadas
        de cada clase. Útil para mostrar una "barra de confianza" en el frontend
        (es el extra opcional que pide el PDF).

        Para convertir log-probabilidades a probabilidades sin caer en underflow,
        usamos el truco de restar el máximo antes de aplicar exp():
            p_i = exp(log_p_i - max_log_p) / Σ exp(log_p_j - max_log_p)
        """
        if not self.entrenado:
            raise RuntimeError("El modelo no ha sido entrenado. Llamar entrenar() primero.")

        log_probs = {}
        for clase in self.clases:
            log_probs[clase] = self._log_probabilidad_clase(tokens, clase)

        # Normalización numéricamente estable (log-sum-exp trick)
        max_log = max(log_probs.values())
        exp_ajustados = {c: math.exp(lp - max_log) for c, lp in log_probs.items()}
        suma = sum(exp_ajustados.values())
        probabilidades = {c: v / suma for c, v in exp_ajustados.items()}

        mejor_clase = max(probabilidades, key=probabilidades.get)
        return mejor_clase, probabilidades

    def predecir_batch(self, lista_documentos):
        """
        Predice una lista completa de documentos.
        Útil para K-Folds cross validation donde necesitamos evaluar muchos ejemplos.
        """
        return [self.predecir(tokens) for tokens in lista_documentos]


# =================================================================
# Bloque de prueba. Se ejecuta solo si corremos este archivo directo.
# =================================================================
if __name__ == "__main__":
    # Entrenamos con un mini-dataset de ejemplo para verificar que todo funciona
    documentos_entrenamiento = [
        ['need', 'cancel', 'order'],
        ['want', 'cancel', 'subscript'],
        ['refund', 'money', 'order'],
        ['ship', 'address', 'wrong'],
        ['track', 'deliveri', 'packag'],
        ['refund', 'money', 'back'],
        ['cancel', 'account', 'pleas'],
        ['ship', 'address', 'chang'],
    ]
    etiquetas_entrenamiento = [
        'CANCEL', 'CANCEL',
        'REFUND', 'SHIPPING',
        'SHIPPING', 'REFUND',
        'CANCEL', 'SHIPPING',
    ]

    modelo = NaiveBayesMultinomial(alpha=1.0)
    modelo.entrenar(documentos_entrenamiento, etiquetas_entrenamiento)

    print(f"Clases aprendidas: {modelo.clases}")
    print(f"Tamaño del vocabulario: {len(modelo.vocabulario)}")
    print(f"Log-priors: {modelo.log_prior}")
    print()

    # Probamos con documentos nuevos
    pruebas = [
        ['want', 'refund', 'money'],
        ['cancel', 'my', 'order'],
        ['where', 'is', 'packag'],
        ['palabra', 'inexistente'],   # caso borde: palabras fuera del vocabulario
    ]

    """
Implementación desde cero del algoritmo Naïve Bayes Multinomial
para clasificación de texto.

No se utilizan librerías de Machine Learning (scikit-learn, etc.).
Todo el cálculo se realiza manualmente usando estructuras nativas de Python.

Conceptos implementados:
    - Bag of Words (vocabulario y conteo de frecuencias por clase)
    - Probabilidades a priori por clase
    - Laplace Smoothing (add-one) para manejar palabras no vistas
    - Suma de logaritmos durante la inferencia para evitar underflow numérico
"""

import math
from collections import defaultdict, Counter


class NaiveBayesMultinomial:
    """
    Clasificador Naïve Bayes Multinomial.

    Funciona con documentos representados como listas de tokens (palabras
    ya preprocesadas). Se entrena con el método entrenar() y predice con
    el método predecir().
    """

    def __init__(self, alpha=1.0):
        """
        Parámetros:
            alpha: parámetro de Laplace Smoothing. Con alpha=1 es el clásico
                   "add-one smoothing". Valores menores hacen menos suavizado.
        """
        self.alpha = alpha

        # Estructuras que se llenan durante el entrenamiento
        self.vocabulario = set()           # conjunto de palabras únicas vistas
        self.clases = []                   # lista de clases posibles
        self.log_prior = {}                # log P(C) para cada clase C
        self.log_verosimilitud = {}        # log P(w|C) para cada palabra w y clase C
        self.conteo_palabras_clase = {}    # total de palabras (con repetición) por clase
        self.entrenado = False

    def entrenar(self, documentos, etiquetas):
        """
        Entrena el modelo a partir de documentos ya preprocesados.

        Parámetros:
            documentos: lista de listas de tokens. Ejemplo:
                        [['need', 'cancel', 'order'], ['refund', 'money'], ...]
            etiquetas: lista de strings con la clase de cada documento.
                       Debe tener la misma longitud que documentos.
        """
        if len(documentos) != len(etiquetas):
            raise ValueError("documentos y etiquetas deben tener la misma longitud")

        n_documentos = len(documentos)
        self.clases = sorted(set(etiquetas))

        # =============================================================
        # PASO 1: Construir el vocabulario global (Bag of Words)
        # =============================================================
        # Recorremos todos los documentos y acumulamos cada palabra única.
        # Este vocabulario se usa en Laplace Smoothing como |V|.
        for tokens in documentos:
            self.vocabulario.update(tokens)

        tamano_vocabulario = len(self.vocabulario)

        # =============================================================
        # PASO 2: Contar documentos y palabras por clase
        # =============================================================
        # documentos_por_clase[C] = cuántos documentos pertenecen a C
        # frecuencia_por_clase[C][w] = cuántas veces aparece w en documentos de C
        # conteo_palabras_clase[C] = suma total de palabras en documentos de C
        documentos_por_clase = defaultdict(int)
        frecuencia_por_clase = defaultdict(Counter)
        self.conteo_palabras_clase = defaultdict(int)

        for tokens, clase in zip(documentos, etiquetas):
            documentos_por_clase[clase] += 1
            for token in tokens:
                frecuencia_por_clase[clase][token] += 1
                self.conteo_palabras_clase[clase] += 1

        # =============================================================
        # PASO 3: Calcular probabilidades a priori log P(C)
        # =============================================================
        # P(C) = (cantidad de documentos de clase C) / (total de documentos)
        # Usamos log() directamente para evitar calcular P y después log P.
        for clase in self.clases:
            prior = documentos_por_clase[clase] / n_documentos
            self.log_prior[clase] = math.log(prior)

        # =============================================================
        # PASO 4: Calcular verosimilitudes log P(w|C) con Laplace Smoothing
        # =============================================================
        # Formula:
        #     P(w|C) = (count(w, C) + alpha) / (total_palabras_C + alpha * |V|)
        #
        # Precalculamos el log de cada P(w|C) para cada palabra del vocabulario
        # en cada clase. Esto hace la inferencia mucho más rápida después.
        self.log_verosimilitud = {clase: {} for clase in self.clases}

        for clase in self.clases:
            # Denominador: total de palabras en la clase + suavizado global
            denominador = self.conteo_palabras_clase[clase] + self.alpha * tamano_vocabulario

            for palabra in self.vocabulario:
                # Numerador: cuántas veces aparece la palabra en la clase + alpha
                numerador = frecuencia_por_clase[clase][palabra] + self.alpha
                self.log_verosimilitud[clase][palabra] = math.log(numerador / denominador)

        self.entrenado = True

    def _log_probabilidad_clase(self, tokens, clase):
        """
        Calcula log P(C|X) ∝ log P(C) + Σ log P(w|C) para cada palabra del documento.

        Notas importantes:
            - Palabras que no están en el vocabulario se ignoran
              (es la convención estándar en Naïve Bayes Multinomial).
            - Palabras que sí están en el vocabulario ya tienen su log P(w|C)
              precalculado con Laplace Smoothing, entonces nunca dan -inf.
        """
        log_prob = self.log_prior[clase]

        for token in tokens:
            if token in self.vocabulario:
                log_prob += self.log_verosimilitud[clase][token]
            # else: palabra nunca vista en entrenamiento -> se ignora

        return log_prob

    def predecir(self, tokens):
        """
        Predice la clase de un documento (lista de tokens preprocesados).
        Retorna el nombre de la clase con mayor log-probabilidad.
        """
        if not self.entrenado:
            raise RuntimeError("El modelo no ha sido entrenado. Llamar entrenar() primero.")

        mejor_clase = None
        mejor_log_prob = -math.inf

        for clase in self.clases:
            log_prob = self._log_probabilidad_clase(tokens, clase)
            if log_prob > mejor_log_prob:
                mejor_log_prob = log_prob
                mejor_clase = clase

        return mejor_clase

    def predecir_con_probabilidades(self, tokens):
        """
        Igual que predecir() pero además retorna las probabilidades normalizadas
        de cada clase. Útil para mostrar una "barra de confianza" en el frontend
        (es el extra opcional que pide el PDF).

        Para convertir log-probabilidades a probabilidades sin caer en underflow,
        usamos el truco de restar el máximo antes de aplicar exp():
            p_i = exp(log_p_i - max_log_p) / Σ exp(log_p_j - max_log_p)
        """
        if not self.entrenado:
            raise RuntimeError("El modelo no ha sido entrenado. Llamar entrenar() primero.")

        log_probs = {}
        for clase in self.clases:
            log_probs[clase] = self._log_probabilidad_clase(tokens, clase)

        # Normalización numéricamente estable (log-sum-exp trick)
        max_log = max(log_probs.values())
        exp_ajustados = {c: math.exp(lp - max_log) for c, lp in log_probs.items()}
        suma = sum(exp_ajustados.values())
        probabilidades = {c: v / suma for c, v in exp_ajustados.items()}

        mejor_clase = max(probabilidades, key=probabilidades.get)
        return mejor_clase, probabilidades

    def predecir_batch(self, lista_documentos):
        """
        Predice una lista completa de documentos.
        Útil para K-Folds cross validation donde necesitamos evaluar muchos ejemplos.
        """
        return [self.predecir(tokens) for tokens in lista_documentos]


# =================================================================
# Bloque de prueba. Se ejecuta solo si corremos este archivo directo.
# =================================================================
if __name__ == "__main__":
    # Entrenamos con un mini-dataset de ejemplo para verificar que todo funciona
    documentos_entrenamiento = [
        ['need', 'cancel', 'order'],
        ['want', 'cancel', 'subscript'],
        ['refund', 'money', 'order'],
        ['ship', 'address', 'wrong'],
        ['track', 'deliveri', 'packag'],
        ['refund', 'money', 'back'],
        ['cancel', 'account', 'pleas'],
        ['ship', 'address', 'chang'],
    ]
    etiquetas_entrenamiento = [
        'CANCEL', 'CANCEL',
        'REFUND', 'SHIPPING',
        'SHIPPING', 'REFUND',
        'CANCEL', 'SHIPPING',
    ]

    modelo = NaiveBayesMultinomial(alpha=1.0)
    modelo.entrenar(documentos_entrenamiento, etiquetas_entrenamiento)

    print(f"Clases aprendidas: {modelo.clases}")
    print(f"Tamaño del vocabulario: {len(modelo.vocabulario)}")
    print(f"Log-priors: {modelo.log_prior}")
    print()

    # Probamos con documentos nuevos
    pruebas = [
        ['want', 'refund', 'money'],
        ['cancel', 'my', 'order'],
        ['where', 'is', 'packag'],
        ['palabra', 'inexistente'],   # caso borde: palabras fuera del vocabulario
    ]

    for tokens in pruebas:
        clase, probs = modelo.predecir_con_probabilidades(tokens)
        probs_formateadas = {k: round(v, 3) for k, v in probs.items()}
        print(f"Documento: {tokens}")
        print(f"  Prediccion: {clase}")
        print(f"  Probabilidades: {probs_formateadas}")
        print()