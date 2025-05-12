# Fraud Detection System

## Overview

This system demonstrates how to use vector embeddings and vector math for anomaly detection in financial transactions. The core concept leverages vector distance calculations to identify potentially fraudulent transactions that deviate significantly from normal patterns.

### How It Works

1. **Vector Representation**: Each transaction is represented as a vector embedding in a high-dimensional space.

2. **Baseline Calculation**: The `get_average_vector.py` script calculates an average vector from all normal transactions in your Astra DB collection, establishing a baseline for "normal" behavior.

3. **Anomaly Detection**: New transactions can be compared against this average vector using cosine similarity or Euclidean distance. Transactions that are mathematically distant from the average are flagged as potential anomalies.

4. **Vector Database**: Astra DB's vector capabilities enable efficient storage and similarity searches across your transaction data.

## Environment Setup

1. Clone this repository
2. Create a `.env` file by copying the sample:
   ```
   cp sample.env .env
   ```
3. Edit the `.env` file and add your Astra DB credentials:
   ```
   ASTRA_TOKEN=your_actual_token_here
   DATABASE_URL=your_actual_database_url_here
   COLLECTION_NAME=your_collection_name
   ```
4. Install the required dependencies:
   ```
   pip install astrapy numpy python-dotenv
   ```

## Usage

### Calculating the Average Vector

Run the script to calculate and update the average vector in your Astra DB collection:

```
python get_average_vector.py
```

### Implementing Anomaly Detection

To detect anomalies in new transactions, you can use the average vector as a reference point:

1. Calculate the vector embedding for a new transaction
2. Compute the distance (cosine or Euclidean) between this vector and the average vector
3. If the distance exceeds a predefined threshold, flag the transaction as a potential anomaly

#### Example Code for Anomaly Detection

```python
import numpy as np
from astrapy import DataAPIClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get credentials from environment variables
astra_token = os.getenv("ASTRA_TOKEN")
database_url = os.getenv("DATABASE_URL")
collection_name = os.getenv("COLLECTION_NAME", "trxdemo")

# Initialize client and database
client = DataAPIClient()
database = client.get_database(database_url, token=astra_token)
collection = database.get_collection(collection_name)

# Fetch the average vector
avg_vector_doc = collection.find_one({"type": "average"})
avg_vector = np.array(avg_vector_doc["$vector"])

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Example: Check if a transaction is anomalous
def check_transaction(transaction_vector, threshold=0.85):
    similarity = cosine_similarity(transaction_vector, avg_vector)
    is_anomalous = similarity < threshold
    return {
        "similarity": similarity,
        "is_anomalous": is_anomalous
    }

# Example usage
# new_transaction_vector = ... # Your new transaction vector
# result = check_transaction(new_transaction_vector)
# print(f"Similarity score: {result['similarity']:.4f}")
# print(f"Anomalous: {result['is_anomalous']}")
```

## Security Note

The `.env` file containing your credentials is included in `.gitignore` to prevent accidentally committing sensitive information to your repository.

## Performance Considerations

- The effectiveness of this approach depends on having a representative dataset of normal transactions
- Consider periodically recalculating the average vector as new normal transactions are added
- Experiment with different distance thresholds to balance false positives and false negatives
- For production use, consider implementing more sophisticated techniques such as clustering or isolation forests alongside vector similarity
