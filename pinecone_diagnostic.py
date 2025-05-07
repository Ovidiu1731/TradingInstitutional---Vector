import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("PINECONE CONNECTION DIAGNOSTIC")
print("-----------------------------")

# Get and display environment variables
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX_NAME") 
host = os.getenv("PINECONE_HOST")

print(f"API Key: {api_key[:4]}{'*' * 10}")
print(f"Environment: {env}")
print(f"Index Name: {index_name}")
print(f"Host (if specified): {host}")
print("\nTesting connection...")

try:
    from pinecone import Pinecone
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    print("✅ Successfully connected to Pinecone API")
    
    # List indexes
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    print(f"✅ Found indexes: {index_names}")
    
    # Check if our target index exists
    if index_name in index_names:
        print(f"✅ Target index '{index_name}' exists")
        
        # Connect to the specific index
        index = pc.Index(index_name)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"✅ Successfully retrieved index stats")
        print(f"   - Vector count: {stats.get('total_vector_count', 0)}")
        
        # Test a dummy query
        try:
            # Get dimension from stats
            dim = 1536  # Default for OpenAI embeddings
            dummy_vector = [0.0] * dim
            
            # Simple query
            results = index.query(vector=dummy_vector, top_k=1, include_metadata=True)
            print(f"✅ Successfully performed test query")
            print(f"   - Returned {len(results.get('matches', []))} results")
        except Exception as e:
            print(f"❌ Test query failed: {str(e)}")
    else:
        print(f"❌ Target index '{index_name}' does not exist!")
        print(f"   Available indexes: {index_names}")
        
except ImportError:
    print("❌ Failed to import Pinecone. Please install with: pip install pinecone")
except Exception as e:
    print(f"❌ Connection error: {str(e)}")

print("\nDiagnostic complete!")
