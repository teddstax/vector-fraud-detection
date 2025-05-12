from astrapy import DataAPIClient
import numpy as np
import os
from dotenv import load_dotenv

def update_average_vector(
    token: str, 
    database_url: str, 
    collection_name: str
):
    try:
        # Initialize client and database
        client = DataAPIClient()
        database = client.get_database(database_url, token=token)
        collection = database.get_collection(collection_name)
        
        # Step 1: Fetch all records that are not "type = 'average'"
        cursor = collection.find(
            filter={"$or": [{"type": None}, {"type": {"$ne": "average"}}, {"type": {"$exists": False}}]},
            projection={"$vector": 1}  # Only fetch the $vector field
        )
        
        # Extract vectors
        embeddings = [doc["$vector"] for doc in cursor if "$vector" in doc]
        
        if not embeddings:
            print("No valid vectors found to calculate an average.")
            return None

        # Step 2: Calculate the new average vector
        embeddings_array = np.array(embeddings)
        new_average_vector = np.mean(embeddings_array, axis=0)

        # Step 3: Delete the existing aggregation record (if it exists)
        collection.delete_many({"type": "average"})
        
        # Step 4: Insert the new aggregation record
        new_record = {
            "type": "average",
            "$vector": new_average_vector.tolist()  # Convert NumPy array to list
        }
        collection.insert_one(new_record)
        
        print("Successfully updated the average vector.")
        return new_average_vector

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get credentials from environment variables
    astra_token = os.getenv("ASTRA_TOKEN")
    database_url = os.getenv("DATABASE_URL")
    collection_name = os.getenv("COLLECTION_NAME", "trxdemo")
    
    # Verify that required environment variables are set
    if not astra_token or not database_url:
        print("Error: Missing required environment variables. Please check your .env file.")
        print("Make sure to copy sample.env to .env and fill in your credentials.")
        exit(1)
        
    avg_vector = update_average_vector(astra_token, database_url, collection_name)
    
    if avg_vector is not None:
        print(f"New average vector calculated. Shape: {avg_vector.shape}")
        print(f"First 5 dimensions of the new average vector: {avg_vector[:5].round(4)}")
