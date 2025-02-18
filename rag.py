import os
from typing import Optional
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
import vertexai
from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
import time
import sys
import json
from google.cloud import storage
from google.api_core import exceptions

# Load environment variables
load_dotenv()

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    project_id: str
    region: str = "us-central1"
    drive_folder_id: str = "1U8TEZKLv3h1F-x9CkNFfG8vimojyYikn"
    embedding_model: str = "publishers/google/models/text-embedding-004"
    llm_model: str = "gemini-2.0-flash-001"  # Updated to match guide
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 10
    vector_distance_threshold: float = 0.5
    service_account_path: Optional[str] = None
    reuse_corpus: bool = True  # New flag
    bucket_name: str = None  # New: GCP bucket for corpus storage
    corpus_prefix: str = "rag-corpora"  # New: Prefix for corpus storage in bucket
    cleanup_corpus: bool = False  # New: Control whether to cleanup corpus
    import_timeout: int = 180  # New: Timeout for import in seconds (3 minutes)
    stable_time: int = 10  # New: Time to wait for stable file count

    def validate(self) -> bool:
        """Validate configuration settings"""
        if not self.project_id:
            raise ValueError("Project ID is required")
        if not self.drive_folder_id:
            raise ValueError("Drive folder ID is required")
        if not self.service_account_path or not os.path.exists(self.service_account_path):
            raise ValueError(f"Service account file not found at: {self.service_account_path}")
        if self.bucket_name is None:
            self.bucket_name = f"{self.project_id}-rag-storage"
        return True

class RAGPipeline:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.rag_corpus = None
        self.llm = None
        self.storage_client = None
        logger.info("Initializing RAG Pipeline...")
        self._initialize_vertex()
        self._initialize_storage()

    def _initialize_vertex(self):
        """Initialize Vertex AI with project settings and authentication"""
        try:
            logger.info("Starting Vertex AI initialization...")
            vertexai.init(
                project=self.config.project_id,
                location=self.config.region,
                credentials=None  # Let Vertex AI handle credentials
            )
            logger.info(f"Project ID: {self.config.project_id}")
            logger.info(f"Region: {self.config.region}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}", exc_info=True)
            raise

    def _initialize_storage(self):
        """Initialize Google Cloud Storage client"""
        try:
            self.storage_client = storage.Client(project=self.config.project_id)
            
            # Try to get existing bucket first
            try:
                bucket = self.storage_client.get_bucket(self.config.bucket_name)
                logger.info(f"Using existing GCS bucket: {bucket.name}")
                return
            except exceptions.NotFound:
                logger.info(f"Bucket {self.config.bucket_name} not found")
            
            # Try to create bucket if it doesn't exist
            try:
                logger.info(f"Attempting to create bucket: {self.config.bucket_name}")
                bucket = self.storage_client.create_bucket(
                    self.config.bucket_name,
                    location=self.config.region
                )
                logger.info(f"Created new GCS bucket: {bucket.name}")
            except exceptions.Forbidden:
                logger.error(
                    "Insufficient permissions to create bucket. Please either:\n"
                    "1. Create the bucket manually and grant storage.objects.* permissions\n"
                    "2. Grant storage.buckets.create permission to the service account"
                )
                raise
                
        except Exception as e:
            logger.error(f"Failed to initialize storage: {str(e)}", exc_info=True)
            raise

    def _get_corpus_blob_name(self, corpus_id: str) -> str:
        """Generate the blob name for corpus storage"""
        return f"{self.config.corpus_prefix}/{corpus_id}/metadata.json"

    def _save_corpus_metadata(self, corpus_name: str):
        """Save corpus metadata to GCS"""
        try:
            metadata = {
                "corpus_name": corpus_name,
                "created_at": time.time(),
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "last_update": time.time(),
                "file_count": len(list(rag.list_files(corpus_name=corpus_name))),
                "drive_folder_id": self.config.drive_folder_id
            }
            
            bucket = self.storage_client.bucket(self.config.bucket_name)
            blob = bucket.blob(self._get_corpus_blob_name(corpus_name.split('/')[-1]))
            blob.upload_from_string(json.dumps(metadata))
            logger.info(f"Saved corpus metadata to: gs://{self.config.bucket_name}/{blob.name}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to save corpus metadata: {str(e)}")
            return None

    def _check_corpus_needs_update(self, corpus_name: str, metadata: dict) -> bool:
        """Check if corpus needs to be updated with new files"""
        try:
            # Add delay to respect rate limits
            time.sleep(1)  # 1 second delay between API calls
            
            # Check if the Drive folder ID has changed
            if metadata.get("drive_folder_id") != self.config.drive_folder_id:
                logger.info("Drive folder ID has changed, corpus needs update")
                return True

            # Get current file count with retry
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    current_files = list(rag.list_files(corpus_name=corpus_name))
                    current_count = len(current_files)
                    stored_count = metadata.get("file_count", 0)

                    if current_count != stored_count:
                        logger.info(f"File count mismatch: stored={stored_count}, current={current_count}")
                        return True

                    return False
                    
                except Exception as e:
                    if "RATE_LIMIT_EXCEEDED" in str(e):
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limit exceeded, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                    raise
                    
        except Exception as e:
            logger.error(f"Error checking corpus update status: {str(e)}")
            # If we can't check, assume no update needed to avoid unnecessary operations
            return False

    def _load_corpus_metadata(self) -> Optional[str]:
        """Load most recent corpus metadata from GCS"""
        try:
            bucket = self.storage_client.bucket(self.config.bucket_name)
            # List blobs with prefix and get the most recent one
            blobs = list(bucket.list_blobs(
                prefix=f"{self.config.corpus_prefix}/",
                max_results=1
            ))
            
            if not blobs:
                return None
                
            # Get the first (and only) blob
            latest_blob = blobs[0]
            metadata = json.loads(latest_blob.download_as_string())
            logger.info(f"Loaded metadata from: {latest_blob.name}")
            return metadata.get("corpus_name")
        except Exception as e:
            logger.error(f"Failed to load corpus metadata: {str(e)}")
            return None

    def setup_corpus(self):
        """Set up the RAG corpus and import documents"""
        try:
            # Check for existing corpus in GCS
            if self.config.reuse_corpus:
                corpus_name = self._load_corpus_metadata()
                if corpus_name:
                    try:
                        self.rag_corpus = rag.get_corpus(name=corpus_name)
                        logger.info(f"Found existing corpus: {corpus_name}")
                        
                        # Load metadata
                        bucket = self.storage_client.bucket(self.config.bucket_name)
                        blob = bucket.blob(self._get_corpus_blob_name(corpus_name.split('/')[-1]))
                        metadata = json.loads(blob.download_as_string())
                        
                        # Check if corpus needs update
                        if not self._check_corpus_needs_update(corpus_name, metadata):
                            logger.info("Corpus is up to date, reusing existing version")
                            return True
                        
                        logger.info("Corpus needs update, importing new files...")
                        
                        # Import new files
                        drive_folder_url = f"https://drive.google.com/drive/folders/{self.config.drive_folder_id}"
                        import_response = rag.import_files(
                            corpus_name=self.rag_corpus.name,
                            paths=[drive_folder_url],
                            chunk_size=self.config.chunk_size,
                            chunk_overlap=self.config.chunk_overlap
                        )
                        logger.info(f"Import response: {import_response}")
                        
                        # Update metadata
                        self._save_corpus_metadata(self.rag_corpus.name)
                        return True
                        
                    except Exception as e:
                        logger.warning(f"Could not reuse corpus {corpus_name}: {str(e)}")
                        logger.info("Creating new corpus...")
                        # Continue to create new corpus

            # Create new corpus if we couldn't reuse existing one
            logger.info("Starting new RAG corpus setup...")
            
            # Create embedding model configuration
            logger.info(f"Configuring embedding model: {self.config.embedding_model}")
            embedding_model_config = rag.EmbeddingModelConfig(
                publisher_model=self.config.embedding_model
            )

            # Create corpus
            logger.info("Creating RAG corpus...")
            start_time = time.time()
            self.rag_corpus = rag.create_corpus(
                display_name="rag-corpus",
                embedding_model_config=embedding_model_config
            )
            
            # Import files from Drive folder
            logger.info(f"Importing files from Drive folder: {self.config.drive_folder_id}")
            drive_folder_url = f"https://drive.google.com/drive/folders/{self.config.drive_folder_id}"
            
            import_response = rag.import_files(
                corpus_name=self.rag_corpus.name,
                paths=[drive_folder_url],
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            logger.info(f"Import response: {import_response}")

            # Wait for import and embedding processing to complete
            logger.info("Waiting for import and embedding processing to complete...")
            self._wait_for_import_completion()
            
            # Save corpus metadata
            self._save_corpus_metadata(self.rag_corpus.name)
            
            logger.info(f"New RAG corpus created successfully in {time.time() - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up RAG corpus: {str(e)}", exc_info=True)
            raise

    def _wait_for_import_completion(self):
        """Wait for corpus import to complete with better timeout handling"""
        wait_start = time.time()
        last_file_count = 0
        stable_count_time = 0
        check_interval = 10  # Check every 10 seconds
        
        logger.info(f"Waiting up to {self.config.import_timeout} seconds for import completion...")
        
        while time.time() - wait_start < self.config.import_timeout:
            try:
                files = list(rag.list_files(corpus_name=self.rag_corpus.name))
                file_count = len(files)
                
                if file_count > 0:
                    elapsed = time.time() - wait_start
                    logger.info(f"[{elapsed:.0f}s] Found {file_count} files in corpus")
                    
                    if file_count == last_file_count:
                        if stable_count_time == 0:
                            stable_count_time = time.time()
                            logger.info(f"File count stable at {file_count}, waiting {self.config.stable_time}s to confirm...")
                        elif time.time() - stable_count_time >= self.config.stable_time:
                            logger.info(f"File count stable at {file_count} for {self.config.stable_time}s")
                            return
                    else:
                        stable_count_time = 0
                        last_file_count = file_count
                        logger.info(f"File count changed to {file_count}, resetting stability timer")
                
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error checking corpus status: {str(e)}")
                time.sleep(check_interval)
                continue
        
        # If we get here, we timed out but have files
        if last_file_count > 0:
            logger.warning(f"Import timed out after {self.config.import_timeout}s, but found {last_file_count} files. Proceeding...")
            return
            
        raise TimeoutError(f"Import timed out after {self.config.import_timeout} seconds with no files")

    def setup_llm(self):
        """Set up the LLM with RAG integration"""
        try:
            logger.info("Setting up LLM...")
            if not self.rag_corpus:
                raise ValueError("RAG corpus not initialized. Call setup_corpus() first.")

            # Create RAG store with appropriate settings
            logger.info("Creating RAG store...")
            rag_store = rag.VertexRagStore(
                rag_corpora=[self.rag_corpus.name],
                similarity_top_k=10,  # Increased from default
                vector_distance_threshold=0.5,  # More permissive threshold
            )

            # Create retrieval tool (without unsupported parameters)
            logger.info("Creating retrieval tool...")
            self.rag_retrieval_tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(source=rag_store)
            )

            # Initialize LLM with RAG integration
            logger.info(f"Initializing LLM model: {self.config.llm_model}")
            self.llm = GenerativeModel(
                self.config.llm_model,
                tools=[self.rag_retrieval_tool],
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            logger.info("LLM setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to set up LLM: {str(e)}", exc_info=True)
            raise

    def query(self, prompt: str) -> str:
        """Query the RAG-enhanced LLM"""
        try:
            logger.info(f"Processing query: {prompt}")
            if not self.llm:
                raise ValueError("LLM not initialized. Call setup_llm() first.")

            # Create a more focused prompt with source attribution request
            enhanced_prompt = f"""Based on the provided context, please answer the following question. When citing sources, use [Document Title] format immediately after the relevant information. If you're not using any specific source for a statement, don't add a citation.

Question: {prompt}

Please provide a clear, concise answer using only the information available in the context. If you can't find relevant information in the context, say so."""

            # Generate response with citations
            response = self.llm.generate_content(
                enhanced_prompt,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                },
                tools=[self.rag_retrieval_tool]
            )
            
            # Track source documents and their Drive IDs
            sources = {}
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                        logger.info("Sources used in response:")
                        for citation in candidate.citation_metadata.citations:
                            source_name = citation.source
                            metadata = citation.metadata if hasattr(citation, 'metadata') else {}
                            file_id = metadata.get('file_id')
                            title = metadata.get('title', source_name)
                            
                            logger.info(f"Source: {source_name}")
                            logger.info(f"File ID: {file_id}")
                            logger.info(f"Title: {title}")
                            
                            # Store source info
                            sources[source_name] = {
                                'file_id': file_id,
                                'title': title
                            }

            # Add source information to response
            response_with_sources = {
                'text': response.text,
                'sources': sources
            }
            
            logger.info(f"Final response with sources: {response_with_sources}")
            return response_with_sources

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}", exc_info=True)
            raise

    def cleanup(self):
        """Clean up RAG resources"""
        try:
            if self.rag_corpus:
                logger.info(f"Cleaning up RAG corpus: {self.rag_corpus.name}")
                # Add delay for resource propagation
                time.sleep(30)  
                rag.delete_corpus(name=self.rag_corpus.name)
                logger.info("RAG corpus cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to cleanup RAG resources: {str(e)}", exc_info=True)
            raise

    def test_query(self):
        """Test the RAG functionality with a sample query."""
        logging.info("Testing RAG query functionality...")
        
        test_queries = [
            "What is Newspeak House and what are its key activities? Please include sources.",
            "What are the important rules or guidelines for guests at Newspeak House? Please cite the relevant documents.",
            "What kind of events happen at Newspeak House? Include document sources."
        ]
        
        try:
            for query in test_queries:
                logging.info(f"\nQuery: {query}")
                response = self.query(query)
                logging.info("Response:")
                logging.info(response)
                logging.info("-" * 80)
            
        except Exception as e:
            logging.error(f"Error during test query: {str(e)}")
            raise

    def list_corpus_files(self):
        """List all files in the corpus with their metadata"""
        try:
            if not self.rag_corpus:
                raise ValueError("RAG corpus not initialized")
                
            logger.info("Listing all files in corpus...")
            files = list(rag.list_files(corpus_name=self.rag_corpus.name))
            
            # Group files by source
            files_by_source = {}
            for file in files:
                source = getattr(file, 'source', 'unknown')
                if source not in files_by_source:
                    files_by_source[source] = []
                files_by_source[source].append(file)
            
            # Print summary
            logger.info(f"Found {len(files)} total files in corpus")
            for source, source_files in files_by_source.items():
                logger.info(f"Source '{source}': {len(source_files)} files")
            
            return files_by_source
            
        except Exception as e:
            logger.error(f"Error listing corpus files: {str(e)}")
            raise

    def cleanup_corpus(self):
        """Clean up the corpus by removing all files and reimporting from Drive"""
        try:
            if not self.rag_corpus:
                raise ValueError("RAG corpus not initialized")
                
            logger.info("Starting corpus cleanup...")
            
            # Delete existing corpus
            logger.info(f"Deleting corpus: {self.rag_corpus.name}")
            rag.delete_corpus(name=self.rag_corpus.name)
            
            # Create new corpus
            logger.info("Creating new corpus...")
            self.rag_corpus = rag.create_corpus(
                display_name="rag-corpus",
                embedding_model_config=rag.EmbeddingModelConfig(
                    publisher_model=self.config.embedding_model
                )
            )
            
            # Import files from Drive
            logger.info(f"Importing files from Drive folder: {self.config.drive_folder_id}")
            drive_folder_url = f"https://drive.google.com/drive/folders/{self.config.drive_folder_id}"
            
            import_response = rag.import_files(
                corpus_name=self.rag_corpus.name,
                paths=[drive_folder_url],
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            logger.info(f"Import response: {import_response}")
            
            # Wait for import to complete
            logger.info("Waiting for import to complete...")
            self._wait_for_import_completion()
            
            # Save new metadata
            self._save_corpus_metadata(self.rag_corpus.name)
            
            # Reinitialize LLM with new corpus
            logger.info("Reinitializing LLM...")
            self.setup_llm()
            
            logger.info("Corpus cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during corpus cleanup: {str(e)}")
            raise

    def _initialize_rag(self):
        """Initialize RAG pipeline"""
        try:
            config = RAGConfig(
                project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
                region=os.environ.get("VERTEX_REGION", "us-central1"),
                service_account_path=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                drive_folder_id=os.environ["DRIVE_FOLDER_ID"],
                cleanup_corpus=False,
                bucket_name=f"{os.environ['GOOGLE_CLOUD_PROJECT']}-rag-storage"
            )
            
            self.rag_pipeline = RAGPipeline(config)
            
            # List current files before setup
            logger.info("Current corpus files before setup:")
            self.list_corpus_files()
            
            self.rag_pipeline.setup_corpus()
            self.rag_pipeline.setup_llm()
            
            # List files after setup
            logger.info("Corpus files after setup:")
            self.list_corpus_files()
            
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise

def main():
    logger.info("Starting RAG pipeline application...")
    
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")
            
        # Load and validate configuration
        config = RAGConfig(
            project_id=project_id,
            region=os.getenv("VERTEX_REGION", "us-central1"),
            service_account_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            drive_folder_id=os.getenv("DRIVE_FOLDER_ID"),
            bucket_name=f"{project_id}-rag-storage",
            cleanup_corpus=False,  # Don't cleanup by default
            import_timeout=180,  # 3 minutes timeout
            stable_time=10  # Wait 10s for stability
        )
        
        # Validate configuration
        config.validate()
        
        # Initialize pipeline
        pipeline = RAGPipeline(config)
        
        # Setup RAG pipeline with progress tracking
        logger.info("Setting up RAG pipeline...")
        setup_start = time.time()
        
        # Setup corpus
        pipeline.setup_corpus()
        corpus_time = time.time() - setup_start
        logger.info(f"Corpus setup completed in {corpus_time:.2f} seconds")
        
        # Setup LLM
        logger.info("Setting up LLM...")
        pipeline.setup_llm()
        total_time = time.time() - setup_start
        logger.info(f"Total setup completed in {total_time:.2f} seconds")
        
        # Run test query
        logger.info("Running test query...")
        pipeline.test_query()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        try:
            if 'pipeline' in locals() and config.cleanup_corpus:
                pipeline.cleanup()
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()