"""
Script para descargar el dataset de Bitext desde Hugging Face
y guardarlo como CSV local.
"""
from datasets import load_dataset
import pandas as pd
import os

print("Descargando dataset de Bitext desde Huggind Face...")
print("(Esto puede tardar un par de minutos la primera vez)")

# Descargar el dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# El dataset viene como DatasetDict, tomamos el split de entreamiento
df = dataset['train'].to_pandas()

# Guardamos solo las columnas que necesitamos: instruccion (text) y category (etiqueta)
df_final = df[['instruction', 'category']]

# Guardamos como CSV en la carpeta dataset/
ruta_salida = os.path.join(os.path.dirname(__file__), 'bitext_dataset.csv')
df_final.to_csv(ruta_salida, index=False, encoding='utf-8')

print(f"\n Dataset guardado en: {ruta_salida}")
print(f"    Total de instancias: {len(df_final)}")
print(f"    Categorias únicas: {df_final['category'].nunique()}")
print(f"\nPrimeras 5 filas:")
print(df_final.head())


