import os
import re
import json
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import threading
from dotenv import load_dotenv

from rag import RAGPipeline, RAGConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_credentials():
    """Setup Google Cloud credentials for Railway deployment"""
    creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not creds:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")
    
    # Check if the value is a JSON string
    try:
        creds_data = json.loads(creds)
        # If we got here, it's valid JSON content
        creds_path = 'google-credentials.json'
        with open(creds_path, 'w') as f:
            json.dump(creds_data, f)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
        logger.info("Successfully wrote Google Cloud credentials from JSON content")
    except json.JSONDecodeError:
        # Not JSON, treat as file path
        if not os.path.exists(creds):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS file not found")
        logger.info("Using existing credentials file")

@dataclass
class ThreadContext:
    """Store context about a thread/conversation"""
    channel_id: str
    thread_ts: Optional[str]
    messages: List[Dict] = None

    def __post_init__(self):
        self.messages = self.messages or []

class RagBot:
    def __init__(self):
        # Setup credentials
        setup_credentials()
        
        # Initialize Slack app
        self.app = App(
            token=os.environ["SLACK_BOT_TOKEN"],
            signing_secret=os.environ["SLACK_SIGNING_SECRET"]
        )
        
        # Initialize RAG pipeline
        self.rag_pipeline = None
        self._initialize_rag()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Thread contexts
        self.thread_contexts: Dict[str, ThreadContext] = {}
        self.context_lock = threading.Lock()

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
            self.rag_pipeline.setup_corpus()
            self.rag_pipeline.setup_llm()
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise

    def _setup_event_handlers(self):
        """Set up Slack event handlers"""
        logger.info("Setting up Slack event handlers")
        # Register async event handlers
        @self.app.event("app_mention")
        async def handle_mention(event, say):
            await self._handle_mention(event, say)
            
        @self.app.event("message")
        async def handle_message(event):
            await self._handle_message(event)
            
        logger.info("Event handlers registered successfully")

    def _get_thread_context(self, channel_id: str, thread_ts: Optional[str] = None) -> ThreadContext:
        """Get or create thread context"""
        context_key = f"{channel_id}:{thread_ts}" if thread_ts else channel_id
        
        with self.context_lock:
            if context_key not in self.thread_contexts:
                self.thread_contexts[context_key] = ThreadContext(
                    channel_id=channel_id,
                    thread_ts=thread_ts
                )
            return self.thread_contexts[context_key]

    async def _handle_mention(self, event, say):
        """Handle when bot is mentioned"""
        try:
            logger.info(f"Received mention event: {event}")
            
            # Get thread context
            thread_ts = event.get("thread_ts", event.get("ts"))
            context = self._get_thread_context(event["channel"], thread_ts)
            logger.info(f"Thread context created for channel: {event['channel']}, thread: {thread_ts}")
            
            # Extract question (remove bot mention)
            text = re.sub(r'<@[^>]+>', '', event["text"]).strip()
            logger.info(f"Extracted question: {text}")
            
            if not text:
                logger.info("Empty question received, sending help message")
                await say(
                    text="Please ask a question! For example: '@Ragbot what is Newspeak House?'",
                    thread_ts=thread_ts
                )
                return
            
            # Build context-aware prompt
            prompt = self._build_prompt(text, context)
            logger.info(f"Built prompt: {prompt}")
            
            # Get response from RAG
            logger.info("Querying RAG pipeline...")
            response = self.rag_pipeline.query(prompt)
            logger.info(f"RAG response received: {response}")
            
            # Send response
            logger.info("Sending response to Slack...")
            await say(
                text=response,
                thread_ts=thread_ts
            )
            logger.info("Response sent successfully")
            
        except Exception as e:
            logger.error(f"Error handling mention: {str(e)}", exc_info=True)
            await say(
                text="Sorry, I encountered an error processing your request. Please try again later.",
                thread_ts=thread_ts
            )

    async def _handle_message(self, event):
        """Handle messages to maintain thread context"""
        try:
            # Skip bot messages
            if event.get("bot_id"):
                logger.debug("Skipping bot message")
                return
                
            logger.info(f"Received message event: {event}")
            
            # Get thread context
            thread_ts = event.get("thread_ts", event.get("ts"))
            context = self._get_thread_context(event["channel"], thread_ts)
            logger.info(f"Updated thread context for channel: {event['channel']}, thread: {thread_ts}")
            
            # Add message to context
            context.messages.append(event)
            
            # Limit context size
            if len(context.messages) > 10:
                context.messages = context.messages[-10:]
            logger.info(f"Thread context updated, current size: {len(context.messages)}")
                
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)

    def _build_prompt(self, question: str, context: ThreadContext) -> str:
        """Build a context-aware prompt"""
        prompt_parts = ["Based on the following context and question, provide a detailed answer:"]
        
        # Add conversation context if available
        if context.messages:
            prompt_parts.append("\nConversation context:")
            for msg in context.messages:
                user = f"<@{msg['user']}>"
                text = msg["text"]
                prompt_parts.append(f"{user}: {text}")
        
        # Add the question
        prompt_parts.append(f"\nQuestion: {question}")
        
        return "\n".join(prompt_parts)

    def start(self):
        """Start the bot"""
        try:
            logger.info("Starting bot in Socket Mode...")
            handler = SocketModeHandler(
                app=self.app,
                app_token=os.environ["SLACK_APP_TOKEN"]
            )
            logger.info("Socket Mode handler created, starting...")
            handler.start()
            logger.info("Bot started successfully in Socket Mode")
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    bot = RagBot()
    bot.start() 