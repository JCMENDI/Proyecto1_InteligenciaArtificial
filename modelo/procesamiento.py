"""
Módulo de preprocesamiento de texto para el clasificador Naïve Bayes.

Realiza las siguientes etapas sobre cada documento:
    1. Limpieza con expresiones regulares (placeholders, URLs, números)
    2. Conversión a minúsculas
    3. Tokenización (separación en palabras)
    4. Eliminación de stopwords (palabras vacías en inglés)
    5. Stemming (reducción de palabras a su raíz)

El resultado es una lista de tokens lista para construir el Bag of Words.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class Preprocesador:
    """
    Clase que encapsula todas las operaciones de limpieza y normalización
    de texto. Se instancia una sola vez y se reutiliza para todos los documentos.
    """

    def __init__(self, idioma='english'):
        # Inicializamos el stemmer de Porter. Es el algoritmo clásico de stemming
        # para inglés, implementado originalmente por Martin Porter en 1980.
        self.stemmer = PorterStemmer()

        # Cargamos las stopwords de NLTK. Son palabras de alta frecuencia que no
        # aportan información semántica relevante para la clasificación
        # (ejemplos: "the", "is", "at", "which", "on").
        self.stopwords = set(stopwords.words(idioma))

        # Patrones regex precompilados (se compilan una sola vez para eficiencia)
        # Placeholder del dataset Bitext: textos del tipo {{Order Number}}
        self.regex_placeholder = re.compile(r'\{\{[^}]+\}\}')
        # URLs
        self.regex_url = re.compile(r'http\S+|www\.\S+')
        # Emails
        self.regex_email = re.compile(r'\S+@\S+')
        # Números (no aportan valor porque son siempre distintos entre tickets)
        self.regex_numeros = re.compile(r'\d+')
        # Cualquier caracter que no sea letra o espacio
        self.regex_especiales = re.compile(r'[^a-zA-Z\s]')
        # Múltiples espacios consecutivos
        self.regex_espacios = re.compile(r'\s+')

    def limpiar(self, texto):
        """
        Aplica limpieza con expresiones regulares. El orden importa:
        primero quitamos los placeholders antes de eliminar las llaves.
        """
        if not isinstance(texto, str):
            return ""

        # 1. Eliminamos placeholders del dataset: {{Order Number}}, {{Account Type}}, etc.
        texto = self.regex_placeholder.sub(' ', texto)

        # 2. Eliminamos URLs y emails (ruido para la clasificación)
        texto = self.regex_url.sub(' ', texto)
        texto = self.regex_email.sub(' ', texto)

        # 3. Convertimos a minúsculas
        texto = texto.lower()

        # 4. Quitamos números
        texto = self.regex_numeros.sub(' ', texto)

        # 5. Quitamos caracteres especiales (puntuación, símbolos)
        texto = self.regex_especiales.sub(' ', texto)

        # 6. Normalizamos espacios múltiples a uno solo
        texto = self.regex_espacios.sub(' ', texto).strip()

        return texto

    def tokenizar(self, texto):
        """
        Separa el texto en una lista de tokens (palabras individuales).
        Usa el tokenizador de NLTK que maneja mejor los casos borde que un simple split().
        """
        return word_tokenize(texto)

    def quitar_stopwords(self, tokens):
        """
        Filtra las palabras vacías de la lista de tokens.
        También descarta tokens de un solo caracter (residuo de la limpieza).
        """
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]

    def aplicar_stemming(self, tokens):
        """
        Reduce cada token a su raíz morfológica usando el algoritmo de Porter.
        Ejemplos:
            running, runs, ran     -> run
            orders, ordered        -> order
            shipping, shipped      -> ship
        Esto unifica variantes de la misma palabra en un solo token,
        reduciendo el tamaño del vocabulario y mejorando la generalización.
        """
        return [self.stemmer.stem(t) for t in tokens]

    def procesar(self, texto):
        """
        Pipeline completo. Aplica todas las etapas en orden y retorna la lista
        final de tokens, lista para ser usada en el Bag of Words.
        """
        texto_limpio = self.limpiar(texto)
        tokens = self.tokenizar(texto_limpio)
        tokens = self.quitar_stopwords(tokens)
        tokens = self.aplicar_stemming(tokens)
        return tokens


# Bloque de prueba rápida. Solo se ejecuta si corremos este archivo directamente
# (no cuando lo importamos desde otro módulo).
if __name__ == "__main__":
    preprocesador = Preprocesador()

    # Ejemplos de prueba tomados del dataset de Bitext
    ejemplos = [
        "I need to change the shipping address for order {{Order Number}}",
        "Help me, I want to cancel my subscription!!! It's been 3 months",
        "Please contact me at support@company.com or visit https://help.com",
        "How can I get a REFUND for my recent purchase?",
    ]

    for i, texto in enumerate(ejemplos, 1):
        print(f"\n--- Ejemplo {i} ---")
        print(f"Original: {texto}")
        print(f"Limpio:   {preprocesador.limpiar(texto)}")
        print(f"Tokens finales: {preprocesador.procesar(texto)}")