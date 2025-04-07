from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get cloud client
cloud_url = os.getenv("QDRANT_URL")
cloud_api_key = os.getenv("QDRANT_API_KEY")

if not cloud_url or not cloud_api_key:
    raise ValueError("Error: QDRANT_URL or QDRANT_API_KEY not found in environment")

cloud_client = QdrantClient(
    url=cloud_url,
    api_key=cloud_api_key
)

# Check collection
try:
    collection_info = cloud_client.get_collection("petrol_transactions")
    print("\nCloud Collection Information:")
    print(f"Points count: {collection_info.points_count}")
    print(f"Vectors size: {collection_info.config.params.vectors.size}")
    print(f"Distance: {collection_info.config.params.vectors.distance}")
    
    if collection_info.points_count > 0:
        # Get a sample point
        points = cloud_client.scroll(
            collection_name="petrol_transactions",
            limit=1,
            with_payload=True,
            with_vectors=True
        )[0]
        if points:
            print("\nSample record:")
            print(f"ID: {points[0].id}")
            print(f"Text: {points[0].payload['text']}")
            print(f"Created at: {points[0].payload['created_at']}")
except Exception as e:
    print(f"Error checking cloud collection: {str(e)}") 