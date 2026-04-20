"""
Servidor Flask que expone el modelo Naive Bayes como API web.

Endpoints:
    GET  /              -> pagina principal (formulario de tickets)
    POST /api/predecir  -> recibe un ticket y devuelve la categoria predicha
    GET  /api/info      -> informacion del modelo (para debug)
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import uuid
from datetime import datetime

# Agregamos la raiz del proyecto al path para poder importar el modulo 'modelo'
# Esto es necesario porque Flask se ejecuta desde la carpeta web/
RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, RAIZ_PROYECTO)

from modelo.entrenar import cargar_modelo


# =====================================================
# Inicializacion de la app y carga del modelo
# =====================================================
# El modelo se carga UNA SOLA VEZ al iniciar el servidor, no en cada request.
# Esto es crucial para el rendimiento: cargar el pickle cada vez seria
# innecesariamente lento.
app = Flask(__name__)

print("Cargando modelo entrenado...")
RUTA_MODELO = os.path.join(RAIZ_PROYECTO, 'modelo', 'modelo_entrenado.pkl')
MODELO, PREPROCESADOR, METADATA = cargar_modelo(RUTA_MODELO)
print(f"Modelo cargado: {METADATA}")


# =====================================================
# Rutas
# =====================================================
@app.route('/')
def index():
    """Sirve la pagina principal con el formulario de tickets."""
    return render_template('index.html')


@app.route('/api/predecir', methods=['POST'])
def predecir():
    """
    Recibe un ticket en JSON y devuelve la categoria predicha + probabilidades.

    Request esperado:
        {
            "subject": "Can't access my account",
            "description": "I forgot my password and need help"
        }

    Response:
        {
            "ticket_id": "TKT-20260420-ABC12",
            "timestamp": "2026-04-20 14:32:01",
            "categoria": "ACCOUNT",
            "confianza": 95.2,
            "probabilidades": { "ACCOUNT": 0.952, "CANCEL": 0.012, ... }
        }
    """
    datos = request.get_json()

    # Validacion basica de entrada
    if not datos:
        return jsonify({'error': 'Body vacio o no es JSON valido'}), 400

    subject = datos.get('subject', '').strip()
    description = datos.get('description', '').strip()

    if not subject and not description:
        return jsonify({'error': 'Debe proveer al menos subject o description'}), 400

    # Combinamos subject y description en un solo texto para clasificar.
    # El subject suele tener palabras clave mas directas, por eso va primero.
    texto_completo = f"{subject} {description}".strip()

    # Pipeline: preprocesar -> clasificar
    tokens = PREPROCESADOR.procesar(texto_completo)
    clase, probabilidades = MODELO.predecir_con_probabilidades(tokens)

    # Generamos ID de ticket con formato: TKT-AAAAMMDD-XXXXX
    # Es autogenerado del lado del backend, como pide el PDF
    ahora = datetime.now()
    ticket_id = f"TKT-{ahora.strftime('%Y%m%d')}-{uuid.uuid4().hex[:5].upper()}"

    respuesta = {
        'ticket_id': ticket_id,
        'timestamp': ahora.strftime('%Y-%m-%d %H:%M:%S'),
        'categoria': clase,
        'confianza': round(probabilidades[clase] * 100, 2),
        'probabilidades': {c: round(p * 100, 2) for c, p in probabilidades.items()},
        'tokens_procesados': tokens,
    }
    return jsonify(respuesta)


@app.route('/api/info')
def info():
    """Devuelve metadata del modelo. Util para debug y para la pagina."""
    return jsonify({
        'n_instancias_entrenamiento': METADATA['n_instancias'],
        'n_clases': METADATA['n_clases'],
        'tamano_vocabulario': METADATA['tamano_vocabulario'],
        'alpha_laplace': METADATA['alpha'],
        'clases': MODELO.clases,
    })


if __name__ == '__main__':
    # debug=True recarga el servidor cuando cambiamos codigo (util en desarrollo)
    app.run(debug=True, host='127.0.0.1', port=5000)