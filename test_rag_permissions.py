import os
from google.oauth2 import service_account
import vertexai
from vertexai.preview import rag
import logging
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def verify_service_account():
    """Verify service account exists and has necessary permissions"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        logger.info(f"Verifying service account for project: {project_id}")
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location="us-central1", credentials=credentials)
        logger.info("✅ Successfully initialized Vertex AI")
        
        # Test RAG corpus creation
        logger.info("Testing RAG corpus creation...")
        embedding_model_config = rag.EmbeddingModelConfig(
            publisher_model="publishers/google/models/text-embedding-004"
        )
        
        test_corpus = rag.create_corpus(
            display_name="test-permissions-corpus",
            embedding_model_config=embedding_model_config
        )
        logger.info(f"✅ Successfully created test corpus: {test_corpus.name}")
        
        # Test file import
        logger.info("Testing file import capabilities...")
        test_file_id = "1WyOkaJSnSEnd_JpZBgr-LLMkSB0cm5fIWfk-tSAXlD4"  # NH Sensors doc
        drive_path = f"https://drive.google.com/file/d/{test_file_id}"
        
        import_response = rag.import_files(
            corpus_name=test_corpus.name,
            paths=[drive_path],
            chunk_size=512,
            chunk_overlap=50
        )
        logger.info("✅ Successfully initiated file import")
        
        # Wait for a bit to check import status
        logger.info("Waiting 30 seconds to check import status...")
        time.sleep(30)
        
        # List files to verify import
        files = list(rag.list_files(corpus_name=test_corpus.name))
        logger.info(f"Found {len(files)} files in corpus after waiting")
        for file in files:
            logger.info(f"Imported file: {file.display_name}")
        
        # Cleanup
        logger.info("Cleaning up test corpus...")
        rag.delete_corpus(name=test_corpus.name)
        logger.info("✅ Successfully cleaned up test corpus")
        
    except Exception as e:
        logger.error(f"❌ Error during verification: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    verify_service_account() 