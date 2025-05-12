# KYC/AML Anomaly Detection Workshop
## A Hands-on Session for Financial Services Professionals

**Duration:** 1 hour  
**Tools:** AstraDB, Langflow, MCP (Model Context Protocol)  
**Focus:** KYC/AML compliance through vector-based anomaly detection

---

## Workshop Overview

In this hands-on workshop, participants will build an ML-powered system to identify suspicious financial transactions for KYC/AML compliance. We'll leverage vector embeddings to establish a baseline for "normal" transactions and identify outliers using distance measurements. When suspicious transactions are flagged, we'll use generative AI to provide human-readable explanations of why the transaction might be fraudulent.

## Prerequisites

- Basic Python knowledge
- Understanding of financial transactions and KYC/AML concepts
- Laptop with internet connection

## Setup Instructions (5 minutes)

1. Create an AstraDB account or use your existing account
2. Set up API credentials for OpenAI or Grok
3. Install Python dependencies
4. Clone the workshop repository

## Workshop Outline

### 1. Introduction (5 minutes)
- Overview of the KYC/AML challenge in financial services
- Introduction to vector similarity for anomaly detection
- Workshop objectives and expected outcomes

### 2. Setting Up AstraDB (10 minutes)
- Create a new database in AstraDB console
- Create a collection for storing transaction data and vectors
- Configure API access and credentials

### 3. Download Custom AstraDB MCP Server (5 minutes)
- Clone the custom MCP server repository: `https://github.com/teddstax/astra-db-mcp-vectorize.git`
- Configure the MCP server with your AstraDB credentials
- Understand the MCP server's capabilities for transaction generation and vector operations

### 4. Use MCP Server to Create Baseline Transactions (10 minutes)
- Use the MCP server in Windsurf or Cursor to create 'good' transactions
- Generate a diverse set of normal financial transactions
- Define transaction attributes (date, time, amount, vendor, location, etc.)
- Store these transactions with vector embeddings in your AstraDB collection

### 5. Calculate Average Vector for Normal Transactions (5 minutes)
- Run the `get_average_vector.py` script to process all 'good' transactions
- Calculate the mathematical average (centroid) of these vectors using NumPy
- Store this centroid vector as our reference point for "normal" behavior

### 6. Generate and Analyze Fraudulent Transactions (15 minutes)
- Use the MCP server to create 'bad' (anomalous) transactions
- Store these potentially fraudulent transactions in AstraDB
- Explore in the AstraDB UI to visualize which transactions are furthest from the average
- Analyze the distance metrics to understand what makes certain transactions suspicious

### 7. Use Langflow for Fraud Explanation (15 minutes)
- Set up a custom Langflow workflow for transaction explanation
- Input several 'good' transactions as context for the LLM
- Ask the LLM to explain what makes the 'bad' transaction different
- Generate human-readable explanations for why transactions were flagged
- Review and refine the explanations

### 8. Conclusion and Next Steps (5 minutes)
- Review what we've built
- Discuss how to further enhance the system (additional models, feedback loops)
- Explore production deployment considerations
- Q&A

## Detailed Implementation Steps

### Generating Synthetic Transactions

We'll use an LLM to generate realistic transaction data with a structured prompt that requests various transaction details including dates, times, amounts, merchants, categories, locations, and payment methods.

### Creating Vector Embeddings

For each transaction, we'll concatenate all fields into a single text string for processing. 
### Calculating the Centroid

We'll use NumPy to calculate the mathematical average (centroid) of all the normal transaction vectors, which will serve as our baseline for comparison.

### Detecting Anomalies

We'll implement functions to calculate the Euclidean distance between new transaction vectors and our baseline centroid. Transactions with distances exceeding our threshold will be flagged as suspicious.

### Explaining with Langflow

In Langflow, we'll create a workflow that:
1. Takes several 'good' transactions as context
2. Takes a 'bad' transaction that was flagged as anomalous
3. Prompts the LLM to explain what makes the bad transaction different from normal ones
4. Returns a formatted explanation that could be used in a compliance report

## Resources and References

- AstraDB Documentation: https://docs.datastax.com/en/astra-db/
- Langflow Documentation: https://docs.langflow.org/
- Vector Similarity Search Guide: https://docs.datastax.com/en/astra-db/docs/vector-search/
- Sample Code Repository: [GitHub Link]

## Frequently Asked Questions (FAQ)

### Technical Questions

**Q: Why use vector embeddings instead of traditional rule-based systems for fraud detection?**  
A: Vector embeddings capture semantic patterns in transaction data that might be missed by rule-based systems. They can identify suspicious activities that don't explicitly violate predefined rules but still deviate from normal patterns. This approach can adapt to evolving fraud tactics without requiring constant rule updates.

**Q: How many "normal" transactions do we need to create a reliable baseline?**  
A: Generally, a few hundred diverse transactions provide a reasonable baseline. For a production system, you'd want thousands across different customer segments. For this workshop, we'll use 50-100 transactions as a starting point.

**Q: What embedding dimension size are we using, and does it matter?**  
A: We're using OpenAI's ada-002 which produces 1536-dimensional vectors. Higher dimensionality allows for more nuanced representation of transaction characteristics, but also requires more computational resources. For production systems, you might experiment with dimension reduction techniques.

**Q: How do we handle different transaction types that might have legitimately different patterns?**  
A: In a production environment, you would create separate centroids for different transaction types, customer segments, or even individual customers. This workshop uses a simpler approach as proof of concept, but we'll discuss these refinements.

**Q: What's the advantage of using Euclidean distance rather than cosine similarity?**  
A: For anomaly detection, Euclidean distance often works better because it considers both the direction and magnitude of differences. Cosine similarity only measures directional differences. However, both metrics can be valuable depending on your specific use case.

### Implementation Questions

**Q: How would we implement this in a production environment with real-time transaction monitoring?**  
A: In production, you would set up a streaming architecture using tools like Kafka or Pulsar that would process transactions in real-time, calculate distances, and trigger alerts based on thresholds. AstraDB supports real-time queries that would make this implementation straightforward.

**Q: Can we create multiple centroids for different customer segments?**  
A: Absolutely! In a production system, you would likely create separate baselines for different customer segments, account types, or even individual customers based on their transaction history.

**Q: How would you handle the cold start problem for new customers?**  
A: New customers can initially be evaluated against a general baseline for their demographic or account type. As they build transaction history, you can gradually shift to a personalized baseline. Some systems use a hybrid approach during this transition period.

**Q: How do we establish appropriate threshold values?**  
A: Threshold determination typically involves analyzing the distribution of distances for known legitimate and fraudulent transactions. A common approach is to start with statistical methods (e.g., setting thresholds at 2 or 3 standard deviations) and then refine based on false positive/negative rates.

**Q: How can we reduce false positives?**  
A: Several approaches include: (1) Using multiple centroids for different transaction types/segments, (2) Implementing a secondary verification system for flagged transactions, (3) Incorporating user feedback to improve thresholds, and (4) Combining vector similarity with traditional rule-based checks.

### AstraDB and Langflow Questions

**Q: How does AstraDB handle vector similarity search at scale?**  
A: AstraDB uses approximate nearest neighbor (ANN) algorithms to efficiently search through millions or billions of vectors. It implements optimized indexing that makes similarity searches extremely fast even with large datasets.

**Q: What's the advantage of using Langflow over direct API calls to an LLM?**  
A: Langflow provides a visual interface for creating complex AI workflows without extensive coding. It makes it easier to experiment with different prompts, add logic between steps, and implement feedback loops. It also helps with prompt versioning and team collaboration.

**Q: What is MCP and why are we using it?**  
A: MCP (Model Context Protocol) is used to connect to AstraDB. In this workshop, we're using it to explore collections and perform bulk operations that would be cumbersome via the standard APIs.

**Q: How much does it cost to run this system in production?**  
A: Costs would depend on transaction volume, embedding model choice, and storage requirements. AstraDB offers various pricing tiers including a free tier for development. For a medium-sized financial institution, costs would typically include database usage, embedding API calls, and LLM usage for explanations.

**Q: Can this system be deployed on-premises for organizations with strict data sovereignty requirements?**  
A: Yes, AstraDB can be deployed on-premises with DataStax Enterprise. The embedding generation and LLM components could use locally-deployed models for organizations that cannot send data to external APIs.

### Business and Compliance Questions

**Q: How does this approach help with regulatory compliance?**  
A: This system provides (1) Automated detection of unusual patterns that might indicate money laundering or fraud, (2) Documented reasoning for why transactions were flagged, which helps with regulatory reporting, and (3) An audit trail of transaction evaluations and decisions.

**Q: How would you integrate this with existing KYC/AML systems?**  
A: This vector-based approach complements traditional rule-based systems. You can implement it alongside existing systems, using it to catch anomalies that rule-based systems might miss. The integration typically happens at the alerting level, potentially using a risk-scoring approach that combines signals from multiple systems.

**Q: How can we explain the ML decision-making process to regulators?**  
A: The combination of distance metrics (quantitative measure of deviation) and LLM-generated explanations provides both technical and human-readable justification for flagging transactions. This transparency helps satisfy regulatory requirements for explainable AI in financial services.

**Q: Can this system detect money laundering patterns that occur across multiple transactions?**  
A: The basic implementation we're building today focuses on individual transactions. However, this approach can be extended to analyze sequences or groups of transactions by creating embeddings that represent transaction patterns over time or across accounts.

**Q: How frequently should we update our "normal" transaction baseline?**  
A: Best practice is to regularly update your baseline (e.g., monthly or quarterly) to account for evolving legitimate behavior patterns. Some systems implement continuous learning where the baseline gradually adapts based on confirmed legitimate transactions.
