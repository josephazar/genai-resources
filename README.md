# RAG System with Azure Document Intelligence

A Retrieval-Augmented Generation (RAG) system that uses Azure Document Intelligence for OCR, Azure OpenAI for embeddings and chat completion, and LanceDB for hybrid search with optional Cohere reranking.

## Features

- **Document OCR**: Uses Azure Document Intelligence Read API ($1/1000 pages)
- **Hybrid Search**: Combines vector and full-text search using LanceDB
- **Reranking**: Optional Cohere reranker for improved relevance
- **Metadata Tracking**: Preserves document filenames as references
- **Batch Processing**: Processes all PDFs in a folder automatically
- **Rate Limit Handling**: Graceful fallback when API limits are reached

## Prerequisites

- Python 3.10+
- Azure OpenAI resource with:
  - Chat model deployment (e.g., gpt-4o-mini)
  - Embedding model deployment (e.g., text-embedding-ada-002)
- Azure Document Intelligence resource
- (Optional) Cohere API key for reranking

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file with your credentials:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_API_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_API_DEPLOYMENT=your-chat-deployment-name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name

# Azure Document Intelligence
DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
DOCUMENT_INTELLIGENCE_KEY=your_key_here

# Cohere (Optional)
COHERE_API_KEY=your_key_here
```

## Usage

### 1. Document Ingestion

Place your PDF documents in the `data/` folder, then run:

```bash
python rag_system.py
```

This will:
- Reset the LanceDB index (removes existing data)
- OCR all PDFs using Azure Document Intelligence
- Create embeddings and store them in LanceDB
- Build a full-text search index

### 2. Testing the System

Run the test script to query the system:

```bash
python test_rag.py
```

This will:
- Load the existing LanceDB index
- Run predefined test questions
- Save results to `rag_test_results.txt`
- Show source documents for each answer

## Project Structure

```
.
├── data/                    # PDF documents to process
├── lancedb/                 # Vector database (created automatically)
├── rag_system.py           # Document ingestion script
├── test_rag.py             # Query testing script
├── requirements.txt        # Python dependencies
├── .env                    # Configuration (create this)
├── rag_test_results.txt    # Test results (created automatically)
└── README.md               # This file
```

## Key Components

### rag_system.py
- `AzureDocumentProcessor`: Handles OCR using Document Intelligence
- `DocumentIngestion`: Manages document processing and indexing
- Resets database on each run for clean ingestion

### test_rag.py
- `RAGTester`: Handles querying and result generation
- Hybrid search with automatic fallback
- Rate limit handling for Cohere API
- Results saved with question, answer, and sources

## API Rate Limits

- **Cohere Trial**: 10 calls/minute (automatic 60s cooldown on limit)
- **Azure Services**: Check your subscription limits

## Customization

### Adding New Questions

Edit the `test_questions` list in `test_rag.py`:

```python
test_questions = [
    "Your question here?",
    # Add more questions...
]
```

### Adjusting Chunk Size

In `rag_system.py`, modify the text splitter:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,      # Adjust size
    chunk_overlap=200,    # Adjust overlap
)
```

### Changing Result Count

In `test_rag.py`, adjust the `k` parameter:

```python
result = self.query(question, k=5)  # Returns top 5 chunks
```

## Troubleshooting

1. **"DeploymentNotFound" Error**: Check your Azure OpenAI deployment names in `.env`
2. **Rate Limit Errors**: System automatically falls back to default ranking
3. **No Documents Processed**: Ensure PDFs are in the `data/` folder
4. **Import Errors**: Activate virtual environment and reinstall requirements

## License

This project is for demonstration purposes. Please ensure compliance with your organization's data policies when processing documents.