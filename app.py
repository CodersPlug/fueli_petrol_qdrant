import streamlit as st
import traceback

# Configure Streamlit page - must be the first st command
st.set_page_config(
    page_title="Análisis de Ventas de Combustible",
    page_icon="⛽",
    layout="wide"
)

from openai import OpenAI
from qdrant_client import QdrantClient

# Collection configuration
COLLECTION_NAME = "petrol_transactions"
VECTOR_SIZE = 3072

# Initialize OpenAI client
@st.cache_resource(show_spinner=False)
def get_openai_client():
    """Get or create a singleton OpenAI client instance."""
    try:
        api_key = str(st.secrets["OPENAI_API_KEY"]).strip()
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error("Error al inicializar el cliente de OpenAI")
        st.error(f"Detalles del error: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        st.stop()

# Initialize Qdrant client with cloud storage
@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    """Get or create a singleton Qdrant client instance."""
    try:
        url = str(st.secrets["QDRANT_URL"]).strip()
        api_key = str(st.secrets["QDRANT_API_KEY"]).strip()
        
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,  # Increased timeout for cloud operations
            prefer_grpc=False,  # Force HTTP
            https=True,  # Force HTTPS
            max_retries=3,  # Add retries
            retry_delay=1,  # 1 second between retries
        )
        
        # Test the connection immediately
        client.get_collections()
        return client
    except Exception as e:
        st.error("Error al conectar con Qdrant Cloud")
        st.error(f"Detalles del error: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        st.stop()

def check_collection_exists():
    """Check if the collection exists and has data."""
    try:
        qdrant = get_qdrant_client()
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
        
        st.sidebar.info(f"Conectado a la base de datos. Registros disponibles: {points_count:,}")
        
        if points_count == 0:
            st.error("La base de datos está vacía. Por favor, ejecute primero el script preprocess_data.py")
            st.stop()
        return True
    except Exception as e:
        st.error("Error al verificar la colección en Qdrant Cloud")
        st.error(f"Detalles del error: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
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
        qdrant = get_qdrant_client()
        
        # Get embedding for the query
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=query,
            dimensions=VECTOR_SIZE
        )
        query_vector = response.data[0].embedding
        
        # Search in Qdrant using search method
        search_result = qdrant.search(
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
        # Check if database is ready
        check_collection_exists()
        
        # Add example questions
        with st.expander("📝 Ejemplos de preguntas"):
            st.markdown("""
            - ¿Cuál es el producto más vendido?
            - ¿Cuántas ventas hubo en enero?
            - ¿Cuál fue la venta más grande?
            - ¿En qué pico se despachó más NS XXI?
            - ¿Cuánto se facturó en total de NS XXI?
            """)
        
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
        st.stop()

if __name__ == "__main__":
    main() 