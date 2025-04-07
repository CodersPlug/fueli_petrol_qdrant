from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import os
from dotenv import load_dotenv
import time
import sys

# Load environment variables
load_dotenv()

# Configuration
COLLECTION_NAME = "petrol_transactions"
VECTOR_SIZE = 3072
BATCH_SIZE = 50  # Reduced batch size for better reliability
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def setup_clients():
    """Setup local and cloud Qdrant clients."""
    try:
        # Local client
        print("Connecting to local Qdrant...")
        local_client = QdrantClient(
            path="./qdrant_data",
            force_disable_check_same_thread=True  # Allow concurrent access
        )
        
        # Cloud client
        print("Connecting to Qdrant Cloud...")
        cloud_url = os.getenv("QDRANT_URL")
        cloud_api_key = os.getenv("QDRANT_API_KEY")
        
        if not cloud_url or not cloud_api_key:
            raise ValueError("Error: QDRANT_URL or QDRANT_API_KEY not found in environment")
        
        cloud_client = QdrantClient(
            url=cloud_url,
            api_key=cloud_api_key,
            timeout=60  # Increased timeout for better reliability
        )
        
        return local_client, cloud_client
    except Exception as e:
        print(f"Error setting up clients: {str(e)}")
        sys.exit(1)

def initialize_cloud_collection(cloud_client):
    """Initialize the collection in Qdrant Cloud."""
    try:
        # Delete existing collection if it exists
        try:
            cloud_client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing cloud collection {COLLECTION_NAME}")
        except Exception:
            pass  # Collection didn't exist
        
        print(f"Creating cloud collection {COLLECTION_NAME}")
        cloud_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            timeout=60
        )
    except Exception as e:
        print(f"Error creating cloud collection: {str(e)}")
        raise

def migrate_batch(cloud_client, points, offset, batch_size):
    """Migrate a batch of points with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            # Get the batch of points
            batch = points[offset:offset + batch_size]
            if not batch:
                return True
            
            # Extract vectors and payloads
            vectors = [point.vector for point in batch]
            payloads = [point.payload for point in batch]
            ids = list(range(offset, offset + len(batch)))
            
            # Upload to cloud
            cloud_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                    for point_id, vector, payload in zip(ids, vectors, payloads)
                ]
            )
            return True
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\nRetrying batch at offset {offset} after error: {str(e)}\n")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\nFailed to migrate batch at offset {offset} after {MAX_RETRIES} attempts: {str(e)}")
                return False

def migrate_data(local_client, cloud_client):
    """Migrate all points from local to cloud."""
    try:
        # Get total points from local
        points = local_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100000,  # Large enough to get all points
            with_vectors=True,
            with_payload=True
        )[0]
        
        total_points = len(points)
        print(f"Total points to migrate: {total_points}")
        
        offset = 0
        successful_points = 0
        
        with tqdm(total=total_points, desc="Migrating records") as pbar:
            while offset < total_points:
                success = migrate_batch(cloud_client, points, offset, BATCH_SIZE)
                if not success:
                    break
                
                batch_size = min(BATCH_SIZE, total_points - offset)
                offset += batch_size
                successful_points += batch_size
                pbar.update(batch_size)
        
        return successful_points
    except Exception as e:
        print(f"\nError during migration: {str(e)}")
        return 0

def verify_migration(local_client, cloud_client):
    """Verify the migration was successful."""
    try:
        local_count = local_client.get_collection(COLLECTION_NAME).points_count
        cloud_count = cloud_client.get_collection(COLLECTION_NAME).points_count
        
        print(f"\nMigration verification:")
        print(f"Local points: {local_count}")
        print(f"Cloud points: {cloud_count}")
        
        if local_count == cloud_count:
            print("✅ Migration successful - all points transferred")
            return True
        else:
            print("⚠️ Migration incomplete - some points were not transferred")
            return False
    except Exception as e:
        print(f"Error verifying migration: {str(e)}")
        return False

def main():
    print("Starting migration to Qdrant Cloud...")
    
    # Setup clients
    local_client, cloud_client = setup_clients()
    
    # Initialize cloud collection
    initialize_cloud_collection(cloud_client)
    
    # Migrate data
    successful_points = migrate_data(local_client, cloud_client)
    
    # Verify migration
    if successful_points > 0:
        verify_migration(local_client, cloud_client)
    else:
        print("\n❌ Migration failed - no points were transferred")

if __name__ == "__main__":
    main() 