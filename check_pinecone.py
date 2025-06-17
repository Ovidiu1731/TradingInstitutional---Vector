import os
from dotenv import load_dotenv
from pinecone import Pinecone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_pinecone_status():
    """Check Pinecone index status and content."""
    # Load environment variables
    load_dotenv()
    PINECONE_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")
    
    try:
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_KEY)
        
        # Check if index exists
        if INDEX_NAME not in pc.list_indexes().names():
            logger.error(f"Index {INDEX_NAME} does not exist!")
            return
        
        # Get index
        index = pc.Index(INDEX_NAME)
        
        # Get index stats
        stats = index.describe_index_stats()
        logger.info(f"\nIndex Statistics:")
        logger.info(f"Total vectors: {stats.total_vector_count}")
        logger.info(f"Dimension: {stats.dimension}")
        
        # Check for specific content
        logger.info("\nChecking for liquidity content in Chapter 6, Lesson 2...")
        results = index.query(
            vector=[0.0] * 1536,  # Dummy vector
            filter={
                "chapter": "Capitolul 6",
                "lesson_number": "02"
            },
            top_k=5,
            include_metadata=True
        )
        
        if results["matches"]:
            logger.info(f"\nFound {len(results['matches'])} vectors for Chapter 6, Lesson 2:")
            for i, match in enumerate(results["matches"], 1):
                logger.info(f"\nMatch {i}:")
                logger.info(f"Score: {match.score}")
                if match.metadata:
                    logger.info(f"Text preview: {match.metadata.get('text', '')[:200]}...")
        else:
            logger.info("No vectors found for Chapter 6, Lesson 2")
            
    except Exception as e:
        logger.error(f"Error checking Pinecone: {e}")

if __name__ == "__main__":
    check_pinecone_status()