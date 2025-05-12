# Fraud Detection System

## Workshop Guide

This repository is part of a KYC/AML Anomaly Detection Workshop that demonstrates how to use vector databases for fraud detection. The complete workshop guide is available in the included files:

- `KYC_AML Anomaly Detection Workshop Using AstraDB and Langflow.pdf`
- `fsi-kyc-aml-workshop.md`

### Workshop Overview

In this hands-on workshop, participants will build an ML-powered system to identify suspicious financial transactions for KYC/AML compliance. We leverage vector embeddings to establish a baseline for "normal" transactions and identify outliers using distance measurements. When suspicious transactions are flagged, we use generative AI to provide human-readable explanations of why the transaction might be fraudulent.

### Workshop Outline

1. **Introduction** - Overview of KYC/AML challenges and vector similarity for anomaly detection
2. **Setting Up AstraDB** - Create a database and collection for storing transaction data and vectors
3. **Download Custom AstraDB MCP Server** - Clone and configure the MCP server for transaction generation
4. **Create Baseline Transactions** - Use the MCP server to generate 'good' transactions
5. **Calculate Average Vector** - Run `get_average_vector.py` to establish the baseline centroid
6. **Generate and Analyze Fraudulent Transactions** - Create 'bad' transactions and explore in AstraDB UI
7. **Use Langflow for Fraud Explanation** - Create workflows to explain why transactions are flagged
8. **Conclusion and Next Steps** - Review and discuss production considerations

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

## Creating Test Transactions

To create 'good' and 'bad' transactions for demonstration purposes, this workshop uses a custom Astra DB MCP (Model Context Protocol) server that leverages the vectorize functionality.

### What is MCP?

MCP (Model Context Protocol) is a standard that connects AI systems with external tools and data sources. In this workshop, we use a custom MCP server to connect to AstraDB, explore collections, and perform bulk operations for generating and managing transaction data.

### MCP Server Setup

1. Clone the MCP server repository:
   ```
   git clone https://github.com/teddstax/astra-db-mcp-vectorize.git
   ```

2. Follow the setup instructions in the MCP server repository to configure and run the server.

3. The MCP server provides specialized functions for:
   - Generating synthetic transaction data that mimics real financial transactions
   - Converting transaction data into vector embeddings using OpenAI's ada-002 model (1536 dimensions)
   - Storing both the transaction data and vector embeddings in your Astra DB collection
   - Creating both normal and anomalous transactions for testing

### Workshop Flow

1. **Set up AstraDB**: Create a database and vector-enabled collection

2. **Generate Normal Transactions**: Use the MCP server to create 50-100 'good' (normal) transactions in your Astra DB collection
   - These transactions establish patterns of legitimate financial activity
   - Each transaction includes attributes like date, time, amount, vendor, location, etc.
   - The MCP server automatically converts these to vector embeddings and stores them in AstraDB

3. **Calculate Baseline**: Run `get_average_vector.py` to calculate the centroid (average vector) of all normal transactions
   - This centroid represents the mathematical center of what constitutes "normal" behavior
   - The script stores this centroid in your AstraDB collection with a special type identifier

4. **Generate Anomalous Transactions**: Use the MCP server to create 'bad' (anomalous) transactions
   - These transactions deviate from normal patterns in subtle or obvious ways
   - They might include unusual amounts, suspicious locations, odd timing, etc.

5. **Detect Anomalies**: Use the anomaly detection code example to identify the fraudulent transactions
   - Calculate the Euclidean distance between each transaction and the baseline centroid
   - Transactions with distances exceeding a threshold are flagged as suspicious

6. **Explain Flagged Transactions**: Use Langflow (optional) to generate human-readable explanations for why transactions were flagged

For detailed step-by-step instructions, refer to the workshop guides included in this repository.
