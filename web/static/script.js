/**
 * Logica del cliente para el sistema de tickets.
 * Conecta el formulario con el backend Flask via fetch.
 */

const API_BASE = '';

// Paleta de colores para cada categoria (para las barras)
const COLORES_CATEGORIA = {
    'ACCOUNT':      '#0d6efd',
    'CANCEL':       '#dc3545',
    'CONTACT':      '#20c997',
    'DELIVERY':     '#fd7e14',
    'FEEDBACK':     '#6f42c1',
    'INVOICE':      '#198754',
    'ORDER':        '#0dcaf0',
    'PAYMENT':      '#ffc107',
    'REFUND':       '#e83e8c',
    'SHIPPING':     '#6610f2',
    'SUBSCRIPTION': '#212529'
};

// Referencias al DOM
const btnEnviar = document.getElementById('btnEnviar');
const subjectInput = document.getElementById('subject');
const descriptionInput = document.getElementById('description');
const charCount = document.getElementById('charCount');
const ticketIdInput = document.getElementById('ticketId');

const estadoInicial = document.getElementById('estadoInicial');
const estadoCargando = document.getElementById('estadoCargando');
const estadoResultado = document.getElementById('estadoResultado');

const categoriaPredicha = document.getElementById('categoriaPredicha');
const confianzaSpan = document.getElementById('confianza');
const idGenerado = document.getElementById('idGenerado');
const barrasContainer = document.getElementById('barrasProbabilidad');
const tokensDebug = document.getElementById('tokensDebug');
const infoModelo = document.getElementById('infoModelo');

// Contador de caracteres del textarea
descriptionInput.addEventListener('input', () => {
    charCount.textContent = descriptionInput.value.length;
});

// Botones de ejemplos rapidos
document.querySelectorAll('.ejemplo').forEach(btn => {
    btn.addEventListener('click', () => {
        subjectInput.value = btn.dataset.subject;
        descriptionInput.value = btn.dataset.desc;
        charCount.textContent = descriptionInput.value.length;
    });
});

// Enviar ticket y clasificar
btnEnviar.addEventListener('click', async () => {
    const subject = subjectInput.value.trim();
    const description = descriptionInput.value.trim();

    if (!subject && !description) {
        alert('Por favor escribe al menos un asunto o descripcion');
        return;
    }

    estadoInicial.classList.add('d-none');
    estadoResultado.classList.add('d-none');
    estadoCargando.classList.remove('d-none');
    btnEnviar.disabled = true;

    try {
        const respuesta = await fetch(`${API_BASE}/api/predecir`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ subject, description })
        });

        if (!respuesta.ok) {
            throw new Error(`HTTP ${respuesta.status}`);
        }

        const datos = await respuesta.json();
        mostrarResultado(datos);

    } catch (error) {
        console.error('Error al clasificar:', error);
        alert('Error al contactar el servidor. Verifica que Flask este corriendo.');
        estadoCargando.classList.add('d-none');
        estadoInicial.classList.remove('d-none');
    } finally {
        btnEnviar.disabled = false;
    }
});

// Renderizar el resultado
function mostrarResultado(datos) {
    estadoCargando.classList.add('d-none');
    estadoResultado.classList.remove('d-none');

    ticketIdInput.value = datos.ticket_id;
    idGenerado.textContent = datos.ticket_id;

    const color = COLORES_CATEGORIA[datos.categoria] || '#6c757d';
    categoriaPredicha.innerHTML = `
        <span class="badge" style="background-color: ${color}">${datos.categoria}</span>
    `;
    confianzaSpan.textContent = `${datos.confianza}%`;

    const probsOrdenadas = Object.entries(datos.probabilidades)
        .sort((a, b) => b[1] - a[1]);

    barrasContainer.innerHTML = '';
    probsOrdenadas.forEach(([categoria, prob]) => {
        const esGanadora = categoria === datos.categoria;
        const colorBarra = COLORES_CATEGORIA[categoria] || '#6c757d';

        const div = document.createElement('div');
        div.className = `barra-probabilidad ${esGanadora ? 'ganadora' : ''}`;
        div.innerHTML = `
            <div class="label-categoria">
                <span class="nombre">${categoria}</span>
                <span class="valor">${prob.toFixed(2)}%</span>
            </div>
            <div class="progress">
                <div class="progress-bar" role="progressbar"
                     style="width: ${prob}%; background-color: ${colorBarra};"
                     aria-valuenow="${prob}" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
        `;
        barrasContainer.appendChild(div);
    });

    tokensDebug.textContent = datos.tokens_procesados.join(', ') || '(ningun token reconocido)';
}

// Cargar info del modelo al inicio
(async () => {
    try {
        const r = await fetch(`${API_BASE}/api/info`);
        const info = await r.json();
        infoModelo.textContent =
            `Modelo entrenado con ${info.n_instancias_entrenamiento.toLocaleString()} tickets | ` +
            `${info.n_clases} categorias | ` +
            `Vocabulario: ${info.tamano_vocabulario.toLocaleString()} palabras`;
    } catch (e) {
        infoModelo.textContent = 'No se pudo conectar al servidor';
    }
})();