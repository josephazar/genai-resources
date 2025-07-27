import os
import shutil
from pathlib import Path
import logging
from dotenv import load_dotenv

# Azure imports
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import AnalyzeResult

# LangChain imports
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain.schema import Document

# LanceDB imports
import lancedb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce Azure SDK verbosity
logging.getLogger("azure").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

class AzureDocumentProcessor:
    def __init__(self):
        self.client = DocumentIntelligenceClient(
            endpoint=os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("DOCUMENT_INTELLIGENCE_KEY"))
        )
    
    def process_document(self, file_path: str) -> str:
        """Process document using Azure Document Intelligence Read API"""
        logger.info(f"Processing document: {file_path}")
        
        try:
            with open(file_path, "rb") as file:
                poller = self.client.begin_analyze_document(
                    model_id="prebuilt-read",
                    body=file
                )
            
            result: AnalyzeResult = poller.result()
            
            # Extract text from all pages
            full_text = ""
            for page in result.pages:
                for line in page.lines:
                    full_text += line.content + "\n"
            
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

class DocumentIngestion:
    def __init__(self):
        # Initialize Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        # Initialize document processor
        self.doc_processor = AzureDocumentProcessor()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize LanceDB path
        self.db_path = "./lancedb"
        
    def reset_database(self):
        """Reset LanceDB by removing existing database"""
        if os.path.exists(self.db_path):
            logger.info(f"Removing existing database at {self.db_path}")
            shutil.rmtree(self.db_path)
        
    def process_documents(self, data_folder: str = "data"):
        """Process all documents in the data folder"""
        # Reset database first
        self.reset_database()
        
        logger.info(f"Processing documents from folder: {data_folder}")
        
        documents = []
        data_path = Path(data_folder)
        
        # Connect to LanceDB after reset
        db = lancedb.connect(self.db_path)
        
        for pdf_file in data_path.glob("*.pdf"):
            logger.info(f"Processing: {pdf_file.name}")
            
            try:
                # OCR the document
                text = self.doc_processor.process_document(str(pdf_file))
                
                # Create chunks
                chunks = self.text_splitter.split_text(text)
                
                # Create Document objects with metadata
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_file.name,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Total document chunks created: {len(documents)}")
        
        # Create vector store
        if documents:
            vector_store = LanceDB.from_documents(
                documents, 
                self.embeddings, 
                connection=db
            )
            
            # Create FTS index for hybrid search
            table = vector_store._table
            table.create_fts_index("text")
            
            logger.info("Vector store created successfully with FTS index")
            logger.info(f"Database saved at: {self.db_path}")
        else:
            logger.warning("No documents were processed")
    
def main():
    # Initialize the ingestion system
    ingestion = DocumentIngestion()
    
    # Process documents
    ingestion.process_documents("data")
    
    print("\n" + "="*80)
    print("Document Ingestion Complete!")
    print("="*80)
    print(f"\nLanceDB index created at: {ingestion.db_path}")
    print("You can now run test_rag.py to query the system.")

if __name__ == "__main__":
    main()