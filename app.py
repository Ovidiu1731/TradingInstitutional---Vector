import os
import re
import json
import logging
import time
import copy
from typing import Dict, Any, Optional, List, Union

# Async libraries
import asyncio
import httpx
import aiohttp
from utils.chunk_filtering import filter_and_rank_chunks
import cachetools

from dotenv import load_dotenv
from utils.query_expansion import expand_query
from collections import deque
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError
import pinecone
from datetime import datetime
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from services.config import get_settings

# Import the improved retrieval function
from improved_retrieval import retrieve_lesson_content

# Add imports at the top for the new functionality
import re
from typing import Tuple

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_requests}/{settings.rate_limit_period}"]
)

# Initialize FastAPI app
app = FastAPI(
    title="Trading Education Assistant API",
    description="AI-powered educational assistant for Trading Institutional course materials",
    version=settings.version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.get("/health")
@limiter.limit("5/minute")
async def health_check(request: Request):
    return {"status": "ok"}

@app.get("/version")
@limiter.limit("5/minute")
async def version(request: Request):
    return {"version": settings.version}

@app.get("/ready")
@limiter.limit("5/minute")
async def ready(request: Request):
    return {"ready": True}

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# ENVIRONMENT & GLOBALS
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", "feedback_log.jsonl")
MIN_SCORE = float(os.getenv("PINECONE_MIN_SCORE", "0.70"))
TOP_K = int(os.getenv("PINECONE_TOP_K", "7"))

# --- Model selection ---
EMBEDDING_MODEL = "text-embedding-ada-002"  # FIXED: Match the vector database
COMPLETION_MODEL = "gpt-4o-mini"  # Switch to mini for better rate limits
TEXT_MODEL = "gpt-4o-mini"  # For text processing

if not (OPENAI_API_KEY and PINECONE_API_KEY):
    logging.error("Missing OpenAI or Pinecone API key(s)")
    raise ValueError("Missing OpenAI or Pinecone API key(s)")

# --- Initialize Async Clients (as per mentor's advice) ---
async_openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        # Add retry mechanism
        transport=httpx.AsyncHTTPTransport(retries=3)
    )
)

# Helper function to verify client before each request
async def ensure_valid_client():
    global async_openai_client
    
    # Check if client needs recreation
    if not async_openai_client:
        logging.warning("OpenAI client was None, recreating...")
        async_openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            http_client=httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                transport=httpx.AsyncHTTPTransport(retries=3)
            )
        )
    return async_openai_client

aiohttp_session: Optional[aiohttp.ClientSession] = None # Initialized at startup

# --- Pinecone Sync Client (use with asyncio.to_thread) ---
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    # Initialize Pinecone with new API
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    logging.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
except Exception as e:
    logging.error(f"Failed to initialize Pinecone client: {e}")
    raise

# --- Conversation History Store (Using TTLCache for auto-eviction) ---
conversation_history = cachetools.TTLCache(maxsize=10000, ttl=86400) # 10k sessions, 24hr TTL

# --- Periodic client refresh task ---
async def refresh_clients_periodically():
    """Periodically refresh the HTTP client connections to avoid stale connections."""
    global async_openai_client, aiohttp_session
    
    while True:
        try:
            await asyncio.sleep(1800)  # 30 minutes
            logging.info("Refreshing client connections...")
            
            # Close existing aiohttp session if it exists
            if aiohttp_session and not aiohttp_session.closed:
                await aiohttp_session.close()
            
            # Create new aiohttp session
            aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                connector=aiohttp.TCPConnector(limit=20, limit_per_host=10)
            )
            
            # Recreate OpenAI client
            if async_openai_client:
                try:
                    await async_openai_client.close()
                except:
                    pass
            
            async_openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                http_client=httpx.AsyncClient(
                    http2=True,
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                    transport=httpx.AsyncHTTPTransport(retries=3)
                )
            )
            
            logging.info("Client connections refreshed successfully")
            
        except Exception as e:
            logging.error(f"Error refreshing clients: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize services and start background tasks on application startup."""
    global aiohttp_session
    
    try:
        # Initialize aiohttp session
        aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=20, limit_per_host=10)
        )
        
        # Start periodic client refresh task
        asyncio.create_task(refresh_clients_periodically())
        
        logger.info("Startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    global aiohttp_session, async_openai_client
    
    if aiohttp_session and not aiohttp_session.closed:
        await aiohttp_session.close()
    
    if async_openai_client:
        try:
            await async_openai_client.close()
        except:
            pass

def normalize_diacritics(text: str) -> str:
    """Normalize Romanian diacritics for better matching."""
    replacements = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T'
    }
    for rom, repl in replacements.items():
        text = text.replace(rom, repl)
    return text

class FeedbackModel(BaseModel):
    session_id: str
    question: str
    answer: str
    feedback: str
    query_type: Optional[str] = "unknown"
    analysis_data: Optional[Dict] = None
    image_url: Optional[str] = None

class TextQuery(BaseModel):
    question: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    chapter: Optional[str] = None
    lesson: Optional[str] = None

@app.get("/ping")
async def ping():
    """Simple ping endpoint for health checks."""
    return {"message": "pong", "timestamp": datetime.now().isoformat()}

def log_feedback(session_id: str, question: str, answer: str, feedback: str,
                 query_type: str, analysis_data: Optional[Dict] = None,
                 image_url: Optional[str] = None) -> bool:
    """Log user feedback to file."""
    try:
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "query_type": query_type,
            "analysis_data": analysis_data,
            "image_url": image_url
        }
        
        # Also log to the new structured file
        os.makedirs("feedback_logs", exist_ok=True)
        
        # Log to both files for compatibility
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
        
        with open("feedback_logs/all_feedback_logs.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
        
        return True
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        return False

def retrieve_relevant_content(question: str, pinecone_results: list) -> str:
    """
    Extract and combine relevant content from Pinecone search results.
    """
    if not pinecone_results:
        return "Nu am găsit informații relevante în materialul cursului."
    
    # Combine the top results
    relevant_texts = []
    seen_content = set()
    
    for result in pinecone_results[:TOP_K]:
        if result.get("metadata") and result["metadata"].get("text"):
            text_content = result["metadata"]["text"]
            
            # Simple deduplication
            content_hash = hash(text_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                
                # Add chapter/lesson context if available
                chapter = result["metadata"].get("chapter", "")
                lesson = result["metadata"].get("lesson", "")
                context_prefix = ""
                if chapter and lesson:
                    context_prefix = f"[{chapter}, Lecția {lesson}] "
                
                relevant_texts.append(f"{context_prefix}{text_content}")
    
    if not relevant_texts:
        return "Nu am găsit informații relevante în materialul cursului."
    
    # Combine all relevant content
    combined_content = "\n\n---\n\n".join(relevant_texts)
    
    # Truncate if too long (keep within token limits)
    max_chars = 8000  # Conservative limit for context
    if len(combined_content) > max_chars:
        combined_content = combined_content[:max_chars] + "\n\n[...conținut trunchiat...]"
    
    return combined_content

@app.post("/feedback")
async def submit_feedback(feedback_data: FeedbackModel):
    """Submit user feedback for the AI responses."""
    try:
        success = log_feedback(
            feedback_data.session_id,
            feedback_data.question,
            feedback_data.answer,
            feedback_data.feedback,
            feedback_data.query_type,
            feedback_data.analysis_data,
            feedback_data.image_url
        )
        return {"success": success, "message": "Feedback logged successfully" if success else "Failed to log feedback"}
    except Exception as e:
        logger.error(f"Error in submit_feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")

@app.get("/admin/export-feedback")
async def export_feedback(request: Request, api_key: str = None):
    """Export all feedback data (admin endpoint)."""
    # Simple API key check
    expected_key = os.getenv("ADMIN_API_KEY", "default-admin-key")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        feedback_data = []
        if os.path.exists(FEEDBACK_LOG):
            with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        feedback_data.append(json.loads(line))
        
        return {"feedback_count": len(feedback_data), "feedback_data": feedback_data}
    except Exception as e:
        logger.error(f"Error exporting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to export feedback")

@app.get("/admin/feedback-analysis")
async def get_feedback_analysis(request: Request, api_key: str = None):
    """Get feedback analysis and statistics (admin endpoint)."""
    expected_key = os.getenv("ADMIN_API_KEY", "default-admin-key")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        from feedback_analyzer import FeedbackAnalyzer
        analyzer = FeedbackAnalyzer(FEEDBACK_LOG)
        analysis = analyzer.analyze_feedback_patterns()
        return analysis
    except Exception as e:
        logger.error(f"Error in feedback analysis: {e}")
        return {"error": str(e)}

@app.get("/admin/generate-improvement-report")
async def generate_improvement_report(request: Request, api_key: str = None):
    """Generate AI improvement report based on feedback (admin endpoint)."""
    expected_key = os.getenv("ADMIN_API_KEY", "default-admin-key")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        from feedback_analyzer import FeedbackAnalyzer
        analyzer = FeedbackAnalyzer(FEEDBACK_LOG)
        report = analyzer.generate_improvement_report()
        return {"report": report}
    except Exception as e:
        logger.error(f"Error generating improvement report: {e}")
        return {"error": str(e)}

@app.post("/admin/optimize-prompts")
async def optimize_system_prompts(request: Request, api_key: str = None):
    """Generate optimized system prompts based on feedback analysis (admin endpoint)."""
    expected_key = os.getenv("ADMIN_API_KEY", "default-admin-key")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        from prompt_optimizer import PromptOptimizer
        optimizer = PromptOptimizer()
        
        # Generate optimized prompt
        optimized_prompt = optimizer.generate_optimized_prompt()
        improvements_summary = optimizer.get_prompt_improvements_summary()
        
        return {
            "optimized_prompt": optimized_prompt,
            "improvements_summary": improvements_summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error optimizing prompts: {e}")
        return {"error": str(e)}

@app.get("/admin/system-health")
async def get_system_health_metrics(request: Request, api_key: str = None):
    """Get system health metrics and performance data (admin endpoint)."""
    expected_key = os.getenv("ADMIN_API_KEY", "default-admin-key")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        # Basic system health metrics
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "pinecone_connection": "healthy" if index else "unhealthy",
            "openai_connection": "healthy" if async_openai_client else "unhealthy",
            "conversation_cache_size": len(conversation_history),
            "conversation_cache_maxsize": conversation_history.maxsize,
            "feedback_log_exists": os.path.exists(FEEDBACK_LOG),
            "environment": {
                "pinecone_index": PINECONE_INDEX_NAME,
                "embedding_model": EMBEDDING_MODEL,
                "completion_model": COMPLETION_MODEL,
                "min_score": MIN_SCORE,
                "top_k": TOP_K
            }
        }
        
        # Add feedback statistics if available
        if os.path.exists(FEEDBACK_LOG):
            feedback_count = 0
            with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
                feedback_count = sum(1 for line in f if line.strip())
            health_data["total_feedback_entries"] = feedback_count
        
        return health_data
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {"error": str(e)}

def identify_query_type(question: str) -> Dict[str, Any]:
    """Identify the type of query to optimize processing."""
    question_lower = question.lower()
    
    # Text-based educational queries
    if any(term in question_lower for term in ["ce este", "cum", "de ce", "explica", "defineste"]):
        return {"type": "educational_definition", "requires_detailed_answer": True}
    
    if any(term in question_lower for term in ["diferenta", "compara", "versus", "vs"]):
        return {"type": "comparison", "requires_detailed_answer": True}
    
    if any(term in question_lower for term in ["exemplu", "exemple", "de exemplu"]):
        return {"type": "example_request", "requires_detailed_answer": True}
    
    if any(term in question_lower for term in ["cand", "quando", "timpul", "ora", "sesiune"]):
        return {"type": "timing_question", "requires_detailed_answer": False}
    
    # Default to general educational query
    return {"type": "general_educational", "requires_detailed_answer": True}

def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON content from text that might contain other content."""
    # Look for JSON-like structures
    json_patterns = [
        r'\{[^{}]*\}',  # Simple JSON objects
        r'\{.*?\}',     # JSON objects with nested content
    ]
    
    for pattern in json_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json_str = match.group(0)
                json.loads(json_str)  # Validate JSON
                return json_str
            except json.JSONDecodeError:
                continue
    
    return None

def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{int(time.time())}_{hash(str(time.time())) % 10000}"

def expand_query(query: str) -> str:
    """Expand query with trading-specific terms."""
    # Use the utility function from utils
    from utils.query_expansion import expand_query as util_expand_query
    return util_expand_query(query)

def analyze_conversation_context(conversation_history: List[Dict], current_question: str) -> Dict[str, Any]:
    """Analyze conversation history to provide better context for the current question."""
    context_info = {
        "has_context": len(conversation_history) > 0,
        "previous_topics": set(),
        "is_follow_up": False,
        "context_clues": [],
        "conversation_summary": "",
        "related_previous_qa": []
    }
    
    if not conversation_history:
        return context_info
    
    # Detect follow-up indicators in Romanian
    follow_up_indicators = [
        "și asta", "și aceasta", "și ala", "și acela", "și aia", "și aceea",
        "ce ziceai", "ce spuneai", "despre ce vorbeam", "acel concept",
        "conceptul ăla", "de care vorbeam", "la fel", "la ce te referi",
        "continuând", "în plus", "mai mult", "și ce", "dar ce",
        "ok dar", "ok și", "bine dar", "bine și", "da dar", "da și"
    ]
    
    current_lower = current_question.lower()
    for indicator in follow_up_indicators:
        if indicator in current_lower:
            context_info["is_follow_up"] = True
            context_info["context_clues"].append(indicator)
    
    # Extract topics and concepts from previous conversation
    trading_concepts = [
        "fvg", "fair value gap", "gap", "gaps", "og", "tg", "tcg", "3g", "3cg", "slg",
        "hod", "lod", "lichiditate", "displacement", "mss", "bos", "choch",
        "sesiune", "sesiuni", "londra", "new york", "dax", "nq", "nasdaq", "us30",
        "gu", "eu", "stop loss", "sl", "tp", "take profit", "be", "break even"
    ]
    
    for exchange in conversation_history[-5:]:  # Look at last 5 exchanges
        question = exchange.get("question", "").lower()
        answer = exchange.get("answer", "").lower()
        
        for concept in trading_concepts:
            if concept in question or concept in answer:
                context_info["previous_topics"].add(concept)
        
        # Store relevant previous Q&A for potential reference
        if len(exchange.get("question", "")) > 10:  # Filter out very short questions
            context_info["related_previous_qa"].append({
                "question": exchange.get("question", "")[:200],  # Truncate long questions
                "answer": exchange.get("answer", "")[:300]       # Truncate long answers
            })
    
    # Create conversation summary for long conversations
    if len(conversation_history) > 3:
        recent_topics = list(context_info["previous_topics"])[:5]  # Top 5 topics
        if recent_topics:
            context_info["conversation_summary"] = f"Conversația anterioară a acoperit: {', '.join(recent_topics)}"
    
    return context_info

def enhance_query_with_context(query: str, context_info: Dict[str, Any]) -> str:
    """Enhance the current query with conversation context for better search."""
    enhanced_query = query
    
    # If it's a follow-up question, add previous topics for better search
    if context_info["is_follow_up"] and context_info["previous_topics"]:
        # Add relevant previous topics to search query
        relevant_topics = list(context_info["previous_topics"])[:3]  # Top 3 most relevant
        enhanced_query += " " + " ".join(relevant_topics)
    
    # Add expanded terms
    expanded = expand_query(query)
    enhanced_query += " " + expanded
    
    return enhanced_query.strip()

def build_context_aware_prompt(query: str, context_info: Dict[str, Any], relevant_content: str) -> str:
    """Build a context-aware prompt that includes conversation history."""
    base_prompt = f"""Întrebare: {query}

Context relevant din cursul Trading Instituțional:
{relevant_content}"""
    
    # Add conversation context if available
    if context_info["has_context"]:
        if context_info["is_follow_up"]:
            base_prompt += f"""

CONTEXT CONVERSAȚIE: Această întrebare pare să fie o continuare a unei discuții anterioare. """
            
            if context_info["conversation_summary"]:
                base_prompt += f"""{context_info["conversation_summary"]}. """
            
            if context_info["related_previous_qa"]:
                base_prompt += f"""

Ultima întrebare discutată: "{context_info["related_previous_qa"][-1]["question"]}"

Te rog să ții cont de contextul conversației și să răspunzi în mod corespunzător, făcând referință la discuția anterioară dacă este relevant."""
        
        elif len(context_info["related_previous_qa"]) > 0:
            base_prompt += f"""

CONTEXT CONVERSAȚIE: Această conversație a mai inclus discuții despre concepte de trading. Te rog să ții cont de contextul general al conversației."""
    
    base_prompt += """

Te rog să răspunzi în română, bazându-te strict pe informațiile din contextul furnizat și ținând cont de conversația anterioară dacă este relevantă."""
    
    return base_prompt

def _build_system_prompt(query_type: str, requires_full_analysis: bool) -> str:
    """Build system prompt based on query type."""
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read()
        
        # Add specific instructions based on query type
        if query_type == "educational_definition":
            base_prompt += "\n\nFocusează-te pe definițiile clare și explicațiile precise din materialul cursului."
        elif query_type == "comparison":
            base_prompt += "\n\nCompară conceptele folosind informațiile din curs și evidențiază diferențele cheie."
        elif query_type == "example_request":
            base_prompt += "\n\nFornizează exemple concrete din materialul cursului dacă sunt disponibile."
        elif query_type == "timing_question":
            base_prompt += "\n\nRăspunde concis la întrebarea despre timp/programare bazându-te pe informațiile din curs."
        
        return base_prompt
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return "You are a helpful trading education assistant for the Trading Institutional program."

@app.post("/ask", response_model=Dict[str, Any])
async def ask_question(query: TextQuery):
    """Main endpoint for text-based educational questions."""
    try:
        question = query.question.strip()
        session_id = query.session_id or generate_session_id()
        
        if not question:
            return {"error": "Question cannot be empty", "session_id": session_id}
        
        logger.info(f"Processing question for session {session_id}: {question[:100]}...")
        
        # Identify query type
        query_analysis = identify_query_type(question)
        query_type = query_analysis["type"]
        
        # Retrieve relevant content from Pinecone
        try:
            # Expand query for better search
            expanded_query = expand_query(question)
            search_query = f"{question} {expanded_query}".strip()
            
            # Use the improved retrieval function
            pinecone_results = retrieve_lesson_content(
                search_query, 
                chapter=query.chapter, 
                lesson=query.lesson, 
                top_k=TOP_K
            )
            
            if not pinecone_results:
                return {
                    "answer": "Nu am găsit informații relevante în materialul cursului pentru această întrebare. Te rog să verifici cu unul dintre mentori sau să întrebi un membru cu mai multă experiență.",
                    "session_id": session_id,
                    "query_type": query_type,
                    "sources": []
                }
            
            # Combine relevant content
            relevant_content = ""
            sources = []
            
            for result in pinecone_results:
                if result.get("text"):
                    relevant_content += result["text"] + "\n\n"
                    sources.append({
                        "chapter": result.get("chapter", "Unknown"),
                        "lesson": result.get("lesson", "Unknown"),
                        "score": result.get("score", 0.0)
                    })
            
            if not relevant_content.strip():
                return {
                    "answer": "Nu am găsit informații relevante în materialul cursului pentru această întrebare.",
                    "session_id": session_id,
                    "query_type": query_type,
                    "sources": []
                }
            
        except Exception as e:
            logger.error(f"Error retrieving content: {e}")
            return {
                "error": "Error retrieving content from knowledge base",
                "session_id": session_id,
                "query_type": query_type
            }
        
        # Get conversation history
        current_history = conversation_history.get(session_id, [])
        if query.conversation_history:
            current_history.extend(query.conversation_history)
        
        # Analyze conversation context for better understanding
        context_info = analyze_conversation_context(current_history, question)
        
        # Enhance search query with conversation context
        enhanced_search_query = enhance_query_with_context(search_query, context_info)
        
        # Re-search with enhanced query if it's a follow-up question
        if context_info["is_follow_up"] and enhanced_search_query != search_query:
            logger.info(f"Enhanced search query for follow-up: {enhanced_search_query}")
            enhanced_pinecone_results = retrieve_lesson_content(
                enhanced_search_query, 
                chapter=query.chapter, 
                lesson=query.lesson, 
                top_k=TOP_K
            )
            
            # Use enhanced results if they're better (more results or higher scores)
            if (len(enhanced_pinecone_results) > len(pinecone_results) or 
                (enhanced_pinecone_results and pinecone_results and 
                 enhanced_pinecone_results[0].get("score", 0) > pinecone_results[0].get("score", 0))):
                pinecone_results = enhanced_pinecone_results
                # Rebuild relevant content with enhanced results
                relevant_content = ""
                sources = []
                for result in pinecone_results:
                    if result.get("text"):
                        relevant_content += result["text"] + "\n\n"
                        sources.append({
                            "chapter": result.get("chapter", "Unknown"),
                            "lesson": result.get("lesson", "Unknown"),
                            "score": result.get("score", 0.0)
                        })
        
        # Build system prompt
        system_prompt = _build_system_prompt(query_type, query_analysis["requires_detailed_answer"])
        
        # Prepare conversation for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 8 exchanges to leave room for context)
        history_to_include = current_history[-8:] if len(current_history) > 8 else current_history
        for exchange in history_to_include:
            if exchange.get("question") and exchange.get("answer"):
                messages.append({"role": "user", "content": exchange.get("question", "")})
                messages.append({"role": "assistant", "content": exchange.get("answer", "")})
        
        # Build context-aware prompt that includes conversation context
        user_message = build_context_aware_prompt(question, context_info, relevant_content)
        messages.append({"role": "user", "content": user_message})
        
        # Get LLM response
        try:
            client = await ensure_valid_client()
            response = await client.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Update conversation history
            current_history.append({"question": question, "answer": answer})
            conversation_history[session_id] = current_history
            
            # Log context information for monitoring
            if context_info["is_follow_up"]:
                logger.info(f"Follow-up question detected for session {session_id}. Context clues: {context_info['context_clues']}")
            if context_info["previous_topics"]:
                logger.info(f"Previous topics in conversation: {list(context_info['previous_topics'])}")
            
            return {
                "answer": answer,
                "session_id": session_id,
                "query_type": query_type,
                "sources": sources[:3],  # Return top 3 sources
                "context_info": {
                    "has_conversation_context": context_info["has_context"],
                    "is_follow_up_question": context_info["is_follow_up"],
                    "previous_topics_count": len(context_info["previous_topics"]),
                    "conversation_length": len(current_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return {
                "error": "Error generating response",
                "session_id": session_id,
                "query_type": query_type
            }
    
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/pinecone-stats")
async def pinecone_stats():
    """Get Pinecone index statistics."""
    try:
        stats = index.describe_index_stats()
        return {
            "index_name": PINECONE_INDEX_NAME,
            "stats": stats,
            "embedding_model": EMBEDDING_MODEL,
            "min_score_threshold": MIN_SCORE,
            "top_k": TOP_K
        }
    except Exception as e:
        logger.error(f"Error getting Pinecone stats: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Test Pinecone connection
        pinecone_healthy = True
        try:
            stats = index.describe_index_stats()
        except Exception as e:
            pinecone_healthy = False
            logger.error(f"Pinecone health check failed: {e}")
        
        # Test OpenAI connection
        openai_healthy = True
        try:
            client = await ensure_valid_client()
            # Simple API test
            test_response = await client.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
        except Exception as e:
            openai_healthy = False
            logger.error(f"OpenAI health check failed: {e}")
        
        return {
            "status": "healthy" if (pinecone_healthy and openai_healthy) else "degraded",
            "services": {
                "pinecone": "healthy" if pinecone_healthy else "unhealthy",
                "openai": "healthy" if openai_healthy else "unhealthy"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }