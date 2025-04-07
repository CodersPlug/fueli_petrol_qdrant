# Análisis de Ventas de Combustible

Una aplicación de análisis de ventas de combustible que utiliza IA para responder preguntas sobre transacciones de venta.

## Características

- Búsqueda semántica de transacciones usando Qdrant Cloud
- Procesamiento de lenguaje natural con GPT-4
- Interfaz intuitiva en español
- Análisis en tiempo real de datos de ventas

## Requisitos

- Python 3.9+
- OpenAI API Key
- Qdrant Cloud account

## Configuración

1. Clonar el repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Configurar variables de entorno en `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "tu-api-key"
   QDRANT_URL = "tu-qdrant-url"
   QDRANT_API_KEY = "tu-qdrant-api-key"
   ```

## Despliegue en Streamlit Cloud

1. Crear una cuenta en [Streamlit Cloud](https://streamlit.io/cloud)
2. Conectar el repositorio de GitHub
3. Configurar las variables de entorno en la configuración del despliegue:
   - OPENAI_API_KEY
   - QDRANT_URL
   - QDRANT_API_KEY
4. Desplegar la aplicación

## Uso Local

```bash
streamlit run app.py
```

## Estructura del Proyecto

- `app.py`: Aplicación principal de Streamlit
- `preprocess_data.py`: Script para procesar datos y cargarlos en Qdrant
- `requirements.txt`: Dependencias del proyecto
- `.streamlit/`: Configuración de Streamlit 