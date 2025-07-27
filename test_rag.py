import os
from typing import List, Dict, Any
import logging
from datetime import datetime
from dotenv import load_dotenv
import time

# Azure imports
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# LanceDB imports
import lancedb
from lancedb.rerankers import CohereReranker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGTester:
    def __init__(self):
        # Initialize Azure OpenAI
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_API_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0
        )
        
        # Connect to existing LanceDB
        self.db_path = "./lancedb"
        if not os.path.exists(self.db_path):
            raise ValueError(f"LanceDB not found at {self.db_path}. Please run rag_system.py first to ingest documents.")
        
        self.db = lancedb.connect(self.db_path)
        
        # Get the table
        tables = self.db.table_names()
        if not tables:
            raise ValueError("No tables found in LanceDB. Please run rag_system.py first.")
        
        self.table = self.db.open_table(tables[0])
        logger.info(f"Connected to LanceDB table: {tables[0]}")
        
        # Initialize Cohere reranker if API key is available
        if os.getenv("COHERE_API_KEY"):
            self.reranker = CohereReranker(api_key=os.getenv("COHERE_API_KEY"))
            logger.info("Cohere reranker initialized")
        else:
            self.reranker = None
            logger.info("Cohere API key not found, using default reranker")
        
        # Track rate limit state
        self.cohere_rate_limited = False
        self.rate_limit_reset_time = 0
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search with optional reranking"""
        logger.info(f"Performing hybrid search for: {query}")
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Check if we should wait for rate limit reset
        if self.cohere_rate_limited and time.time() < self.rate_limit_reset_time:
            wait_time = self.rate_limit_reset_time - time.time()
            if wait_time > 0:
                logger.info(f"Waiting {wait_time:.1f}s for Cohere rate limit reset...")
                time.sleep(wait_time)
                self.cohere_rate_limited = False
        
        # Try with reranker first if available
        try:
            if self.reranker and not self.cohere_rate_limited:
                results = (
                    self.table.search(query_type="hybrid")
                    .vector(query_embedding)
                    .text(query)
                    .limit(k * 2)  # Get more results for reranking
                    .rerank(self.reranker)
                    .limit(k)
                    .to_pandas()
                )
                logger.info("Successfully used Cohere reranker")
            else:
                # Fallback to default reranker
                results = (
                    self.table.search(query_type="hybrid")
                    .vector(query_embedding)
                    .text(query)
                    .limit(k)
                    .to_pandas()
                )
        except Exception as e:
            # If reranker fails (e.g., rate limit), fallback to default
            if "429" in str(e) or "Too Many Requests" in str(e):
                logger.warning("Cohere rate limit hit (10 calls/minute for trial), falling back to default reranker")
                # Set rate limit flag and reset time (wait 60 seconds)
                self.cohere_rate_limited = True
                self.rate_limit_reset_time = time.time() + 60
            else:
                logger.warning(f"Reranker error: {str(e)}, falling back to default reranker")
            
            # Fallback search without reranker
            results = (
                self.table.search(query_type="hybrid")
                .vector(query_embedding)
                .text(query)
                .limit(k)
                .to_pandas()
            )
        
        # Format results
        formatted_results = []
        
        # Log available columns for debugging (only once)
        if len(results) > 0 and not hasattr(self, '_columns_logged'):
            logger.info(f"Available columns in search results: {list(results.columns)}")
            self._columns_logged = True
        
        for _, row in results.iterrows():
            # Extract source from metadata dictionary
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                source = metadata.get("source", "Unknown")
            else:
                source = "Unknown"
            
            formatted_results.append({
                "content": row["text"],
                "source": source,
                "score": row.get("_relevance_score", 0.0)
            })
        
        return formatted_results
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query the RAG system and get an answer with metadata"""
        logger.info(f"Processing question: {question}")
        
        # Get relevant documents
        search_results = self.hybrid_search(question, k=k)
        
        # Build context
        context = "\n\n".join([
            f"Source: {result['source']}\nContent: {result['content']}"
            for result in search_results
        ])
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question. 
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        # Get answer from LLM
        response = self.llm.invoke(prompt)
        
        # Get unique sources
        sources = list(set([result['source'] for result in search_results]))
        
        return {
            "question": question,
            "answer": response.content,
            "sources": sources,
            "num_chunks_retrieved": len(search_results)
        }
    
    def test_questions(self, questions: List[str], output_file: str = "rag_test_results.txt"):
        """Test multiple questions and save results to file"""
        results = []
        
        print("\n" + "="*80)
        print("Starting RAG System Testing")
        print("="*80 + "\n")
        
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question}")
            try:
                result = self.query(question)
                results.append(result)
                print(f"✓ Completed\n")
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "num_chunks_retrieved": 0
                })
                print(f"✗ Failed\n")
        
        # Save results to file
        self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save test results to a text file (overwrites existing file)"""
        # Ensure we're overwriting the file by using 'w' mode
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RAG System Test Results\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Summary statistics
            successful_queries = [r for r in results if "Error:" not in r['answer']]
            failed_queries = [r for r in results if "Error:" in r['answer']]
            
            f.write(f"Total Questions: {len(results)}\n")
            f.write(f"Successful: {len(successful_queries)}\n")
            f.write(f"Failed: {len(failed_queries)}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Question {i}: {result['question']}\n")
                f.write("-"*40 + "\n")
                f.write(f"Answer: {result['answer']}\n\n")
                f.write(f"Sources: {', '.join(result['sources']) if result['sources'] else 'None'}\n")
                f.write(f"Chunks Retrieved: {result['num_chunks_retrieved']}\n")
                f.write("\n" + "="*80 + "\n\n")
        
        logger.info(f"Results saved to {output_file} (file overwritten)")
        print(f"\nTest results saved to: {output_file}")
        print(f"Summary: {len(successful_queries)}/{len(results)} queries completed successfully")

def main():
    # Initialize the RAG tester
    tester = RAGTester()
    
    # Define test questions
    test_questions = [
        "What is the employee resignation policy?",
        "How does the performance appraisal process work?",
        "What are the expense claim procedures?",
        "What is the probation period policy?",
        "How does the 360 feedback system work?",
        "What are the transportation claim policies?",
        "Explain the sales commission structure",
        "What is the procedure for job codes and time logging?",
        "What happens during employee onboarding?",
        "What constitutes misconduct according to the policy?",
        "How does the PIP (Performance Improvement Plan) process work?",
        "What are the budgeting guidelines?",
        "What is the travel policy for employees?",
        "How does the client payment collection process work?",
        "What are the attendance and leave policies?"
    ]
    
    # Run tests
    tester.test_questions(test_questions)

if __name__ == "__main__":
    main()