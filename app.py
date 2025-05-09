import streamlit as st
import traceback
import requests
import json
import httpx

# Configure Streamlit page - must be the first st command
st.set_page_config(
    page_title="Análisis de Ventas de Combustible",
    page_icon="⛽",
    layout="wide"
)

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Collection configuration
COLLECTION_NAME = "petrol_transactions"
VECTOR_SIZE = 3072

# Initialize OpenAI client
@st.cache_resource(show_spinner=False)
def get_openai_client():
    """Initialize OpenAI client with error handling."""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("Error: No se encontró la clave API de OpenAI")
            st.stop()
        
        # Create a custom HTTP client without any proxy settings
        http_client = httpx.Client(
            base_url="https://api.openai.com/v1",
            timeout=60.0,
            follow_redirects=True
        )
        
        # Initialize OpenAI with custom client
        return OpenAI(
            api_key=api_key,
            http_client=http_client
        )
        
    except Exception as e:
        st.error("Error al inicializar el cliente de OpenAI")
        st.error(f"Detalles del error: {str(e)}")
        st.stop()

def get_collection_info_direct(url: str, api_key: str) -> dict:
    """Get collection info directly using requests to handle version mismatches."""
    try:
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }
        response = requests.get(
            f"{url}/collections/{COLLECTION_NAME}",
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Looking at the actual response format we're getting:
        # {"result":{"status":"green","optimizer_status":"ok","indexed_vectors_count":82100,"points_count":82311,...}}
        if isinstance(data, dict):
            result = data.get("result", {})
            points_count = result.get("points_count", 0)
            if points_count == 0:
                # Try alternative fields if points_count is 0
                points_count = result.get("indexed_vectors_count", 0)
            
            return {
                "vectors_count": points_count,
                "config": result.get("config", {})
            }
        else:
            st.error("Formato de respuesta inesperado")
            st.error(f"Respuesta del servidor: {data}")
            st.stop()
            
    except requests.exceptions.RequestException as e:
        st.error("Error al obtener información de la colección")
        st.error(f"Error de conexión: {str(e)}")
        st.stop()
    except Exception as e:
        st.error("Error inesperado al obtener información de la colección")
        st.error(f"Detalles del error: {str(e)}")
        if 'response' in locals():
            st.error(f"Respuesta del servidor: {response.text}")
        st.stop()

# Initialize Qdrant client with cloud storage
@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    """Get or create a singleton Qdrant client instance."""
    try:
        url = st.secrets.get("QDRANT_URL")
        api_key = st.secrets.get("QDRANT_API_KEY")
        
        if not url or not api_key:
            st.error("Error: No se encontraron las credenciales de Qdrant")
            st.stop()
        
        # Initialize with explicit parameters
        client_params = {
            "url": url,
            "api_key": api_key,
            "timeout": 60,  # Increased timeout for cloud operations
            "prefer_grpc": False  # Force HTTP
        }
        return QdrantClient(**client_params)
        
    except Exception as e:
        st.error("Error al conectar con Qdrant Cloud")
        st.error(f"Detalles del error: {str(e)}")
        st.stop()

def check_collection_exists():
    """Check if the collection exists and has data."""
    try:
        url = str(st.secrets["QDRANT_URL"]).strip()
        api_key = str(st.secrets["QDRANT_API_KEY"]).strip()
        
        # Try to get collection info directly first
        collection_info = get_collection_info_direct(url, api_key)
        
        if not collection_info:
            st.error("No se pudo obtener información de la colección")
            st.stop()
            
        points_count = collection_info.get("vectors_count", 0)
        
        if points_count == 0:
            st.error("La base de datos está vacía. Por favor, ejecute primero el script preprocess_data.py")
            st.stop()
            
        # Initialize the client only after we confirm the collection exists
        global _qdrant_client
        _qdrant_client = get_qdrant_client()
        
        return points_count
    except Exception as e:
        st.error("Error al verificar la colección en Qdrant Cloud")
        st.error(f"Detalles del error: {str(e)}")
        st.stop()

def get_answer_from_gpt(query: str, context: list[str]) -> str:
    """Get a concise answer from GPT based on the search results."""
    try:
        client = get_openai_client()
        system_message = """Eres un experto analista de datos de ventas de combustible.
        Tu tarea es analizar los datos proporcionados y responder preguntas sobre las ventas.
        IMPORTANTE:
        - Responde de manera CONCISA y DIRECTA en una sola línea
        - NO des explicaciones ni contexto adicional
        - Si la pregunta requiere análisis de totales o tendencias, calcula los números exactos
        - Responde siempre en español"""
        
        context_text = "\n".join([f"- {text}" for text in context])
        user_message = f"""Pregunta: {query}

Datos relevantes:
{context_text}

Responde de manera concisa y directa:"""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar respuesta: {str(e)}")
        st.stop()

def search_similar_transactions(query: str, limit: int = 10):
    """Search for similar transactions using the query text."""
    try:
        client = get_openai_client()
        
        # Get embedding for the query
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=query,
            dimensions=VECTOR_SIZE
        )
        query_vector = response.data[0].embedding
        
        # Search in Qdrant using search method
        search_result = _qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        
        return search_result
    except Exception as e:
        st.error(f"Error al buscar transacciones similares: {str(e)}")
        st.stop()

def main():
    st.title("🔍 Análisis de Ventas de Combustible")
    
    try:
        # Check if database is ready and get record count
        points_count = check_collection_exists()
        
        # Create two columns for the header area
        col1, col2 = st.columns([3, 1])
        
        # Add example questions in the left column
        with col1:
            with st.expander("📝 Ejemplos de preguntas"):
                st.markdown("""
                - ¿Cuál es el producto más vendido?
                - ¿Cuántas ventas hubo en enero?
                - ¿Cuál fue la venta más grande?
                - ¿En qué pico se despachó más NS XXI?
                - ¿Cuánto se facturó en total de NS XXI?
                """)
        
        # Add database status in the right column
        with col2:
            st.metric(
                label="Registros en la base de datos",
                value=f"{points_count:,}"
            )
        
        # Search interface
        query = st.text_input("💭 Ingrese su pregunta:", placeholder="Ejemplo: ¿Cuál es el producto más vendido?")
        
        if query:
            with st.spinner("Analizando..."):
                # Get relevant transactions
                results = search_similar_transactions(query)
                
                if results:
                    # Extract texts from results
                    relevant_texts = [hit.payload['text'] for hit in results]
                    
                    # Get answer from GPT
                    answer = get_answer_from_gpt(query, relevant_texts)
                    
                    # Display answer
                    st.success(answer)
                else:
                    st.warning("No se encontraron resultados relevantes.")
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")

if __name__ == "__main__":
    main() 