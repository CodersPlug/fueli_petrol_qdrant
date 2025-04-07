import streamlit as st

# Configure Streamlit page - must be the first st command
st.set_page_config(
    page_title="Análisis de Ventas de Combustible",
    page_icon="⛽",
    layout="wide"
)

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Get or create a singleton OpenAI client instance."""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in secrets")
        
        try:
            # First try the new style initialization
            return OpenAI(api_key=api_key)
        except TypeError:
            # If that fails, try importing the older version
            import openai
            openai.api_key = api_key
            return openai
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.error("Please check that openai package is installed correctly")
        st.stop()

# Get the OpenAI client instance
client = get_openai_client()

# Initialize Qdrant client with cloud storage
@st.cache_resource
def get_qdrant_client():
    """Get or create a singleton Qdrant client instance."""
    try:
        return QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"]
        )
    except Exception as e:
        st.error(f"Error connecting to Qdrant Cloud: {str(e)}")
        st.stop()

# Collection configuration
COLLECTION_NAME = "petrol_transactions"
VECTOR_SIZE = 3072

def check_collection_exists():
    """Check if the collection exists and has data."""
    qdrant = get_qdrant_client()
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
        if points_count == 0:
            st.error("La base de datos está vacía. Por favor, ejecute primero el script preprocess_data.py")
            st.stop()
        return True
    except Exception as e:
        st.error("Error: La base de datos no está inicializada. Por favor, ejecute primero el script preprocess_data.py")
        st.stop()

def get_answer_from_gpt(query: str, context: list[str]) -> str:
    """Get a concise answer from GPT based on the search results."""
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

def search_similar_transactions(query: str, limit: int = 10):
    """Search for similar transactions using the query text."""
    try:
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