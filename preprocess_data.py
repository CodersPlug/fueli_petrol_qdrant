import pandas as pd
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
import os
from tqdm import tqdm
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Configuration
COLLECTION_NAME = "petrol_transactions"
VECTOR_SIZE = 3072
BATCH_SIZE = 10

def convert_numeric(value):
    """Convert string number with comma as decimal separator to float."""
    if pd.isna(value):
        return np.nan
    # Replace the last comma with a dot for decimal point
    parts = str(value).rsplit(',', 1)
    if len(parts) == 2:
        # Remove any dots (thousand separators) and join with decimal point
        return float(parts[0].replace('.', '') + '.' + parts[1])
    return float(value.replace('.', ''))

def setup_clients():
    """Setup OpenAI and Qdrant clients with error handling."""
    try:
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Error: OPENAI_API_KEY not found in environment")
        openai_client = OpenAI(api_key=api_key)
        
        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("Error: QDRANT_URL or QDRANT_API_KEY not found in environment")
        
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        return openai_client, qdrant_client
    except Exception as e:
        print(f"Error setting up clients: {str(e)}")
        sys.exit(1)

def process_csv():
    """Load and process the CSV file."""
    try:
        if not os.path.exists('Petrol - Despachos.csv'):
            raise FileNotFoundError("CSV file 'Petrol - Despachos.csv' not found")
        
        df = pd.read_csv('Petrol - Despachos.csv')
        
        # Convert numeric columns
        df['Importe'] = df['Importe'].apply(convert_numeric)
        df['Volumen'] = df['Volumen'].apply(convert_numeric)
        # PPU is already numeric
        
        processed_data = []
        for _, row in df.iterrows():
            text = f"Venta de {row['Producto']} por {row['Volumen']:.2f} galones en el pico {row['Pico']} el {row['Fecha y Hora']}. Importe: ${row['Importe']:.2f}, PPU: ${float(row['PPU']):.2f}"
            processed_data.append({
                "text": text,
                "created_at": row['Fecha y Hora']
            })
        return processed_data
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        sys.exit(1)

def initialize_collection(qdrant_client):
    """Initialize the Qdrant collection if it doesn't exist."""
    try:
        # Delete existing collection if it exists
        try:
            qdrant_client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection {COLLECTION_NAME}")
        except Exception:
            pass  # Collection didn't exist
        
        print(f"Creating collection {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        sys.exit(1)

def populate_collection(openai_client, qdrant_client, processed_data):
    """Populate the collection with vectors."""
    try:
        total_batches = len(processed_data) // BATCH_SIZE + (1 if len(processed_data) % BATCH_SIZE else 0)
        
        with tqdm(total=len(processed_data), desc="Processing records") as pbar:
            for i in range(0, len(processed_data), BATCH_SIZE):
                batch = processed_data[i:i + BATCH_SIZE]
                
                try:
                    # Get embeddings for the batch
                    texts = [item['text'] for item in batch]
                    response = openai_client.embeddings.create(
                        model="text-embedding-3-large",
                        input=texts,
                        dimensions=VECTOR_SIZE
                    )
                    embeddings = [e.embedding for e in response.data]
                    
                    # Prepare points for insertion
                    points = [
                        models.PointStruct(
                            id=i + idx,
                            vector=embedding,
                            payload={
                                "text": item['text'],
                                "created_at": item['created_at']
                            }
                        )
                        for idx, (item, embedding) in enumerate(zip(batch, embeddings))
                    ]
                    
                    # Insert points
                    qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    
                    pbar.update(len(batch))
                except Exception as e:
                    print(f"\nError processing batch {i//BATCH_SIZE + 1}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error populating collection: {str(e)}")
        sys.exit(1)

def main():
    print("Starting data preprocessing...")
    
    # Setup clients
    openai_client, qdrant_client = setup_clients()
    
    # Initialize collection
    initialize_collection(qdrant_client)
    
    # Process CSV data
    print("Processing CSV file...")
    processed_data = process_csv()
    print(f"Processed {len(processed_data)} records")
    
    # Populate collection
    print("Populating Qdrant collection...")
    populate_collection(openai_client, qdrant_client, processed_data)
    
    print("Done! Database is ready for use.")

if __name__ == "__main__":
    main() 