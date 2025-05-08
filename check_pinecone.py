import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Get configuration
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "trading-lessons")

print(f"Connecting to Pinecone index: {index_name}")
print(f"Using API key: {api_key[:4]}***")

try:
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    print("✅ Connected to Pinecone")
    
    # List available indexes
    indexes = pc.list_indexes()
    print(f"✅ Found {len(indexes)} indexes in your account:")
    for idx in indexes:
        print(f"- {idx.name}")
    
    # Connect to specific index
    index = pc.Index(index_name)
    print(f"✅ Connected to index: {index_name}")
    
    # Get stats
    stats = index.describe_index_stats()
    print("✅ Index stats:")
    print(f"- Total vectors: {stats.get('total_vector_count', 'N/A')}")
    
except Exception as e:
    print(f"❌ Error: {e}")